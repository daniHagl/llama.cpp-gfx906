#include "common-checkpoint-db.h"
#include "common.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

// --- manifest binary format (version 2) ---
// uint32_t magic        = 0xDBDBDBDB
// uint32_t version      = 2
// uint32_t model_id_len  // number of bytes in model_id string (0 = no model keying)
// uint8_t[model_id_len] // model fingerprint string
// uint32_t n_entries
// for each entry:
//   uint32_t id
//   uint64_t n_tokens
//   llama_token[n_tokens]
//   uint64_t kv_main_size
//   uint64_t kv_drft_size
//   uint32_t n_checkpoints
//   for each checkpoint:
//     int64_t  ckpt_n_tokens
//     int32_t  ckpt_pos_min
//     int32_t  ckpt_pos_max
//     uint64_t ckpt_tgt_size
//     uint64_t ckpt_dft_size
//   uint64_t disk_offset
//   uint32_t disk_size
//
// version 1 manifests (no model_id) are discarded on load — unsafe to reuse
// across different models.

static const uint32_t MANIFEST_MAGIC   = 0xDBDBDBDB;
static const uint32_t MANIFEST_VERSION = 2;

// helpers
static void write_u32(std::ofstream & os, uint32_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static void write_u64(std::ofstream & os, uint64_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static void write_i64(std::ofstream & os, int64_t  v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static void write_i32(std::ofstream & os, int32_t  v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }

static uint32_t read_u32(std::ifstream & is) { uint32_t v; is.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
static uint64_t read_u64(std::ifstream & is) { uint64_t v; is.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
static int64_t  read_i64(std::ifstream & is) { int64_t  v; is.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
static int32_t  read_i32(std::ifstream & is) { int32_t  v; is.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }

checkpoint_db::checkpoint_db(const checkpoint_db_config & cfg)
    : cfg_(cfg)
    , root_(std::make_unique<trie_node>())
{
}

checkpoint_db::~checkpoint_db() {
    try {
        flush();
    } catch (...) {}
}

std::string checkpoint_db::manifest_path() const {
    return cfg_.disk_path + "/manifest.bin";
}

std::string checkpoint_db::kv_path(uint32_t id) const {
    return cfg_.disk_path + "/e/" + std::to_string(id) + ".kv";
}

void checkpoint_db::load_manifest() {
    if (cfg_.disk_path.empty()) {
        return;
    }

    const auto mpath = manifest_path();
    std::ifstream is(mpath, std::ios::binary);
    if (!is.is_open()) {
        LOG_INF("checkpoint_db: no manifest at %s, starting fresh\n", mpath.c_str());
        return;
    }

    const auto magic = read_u32(is);
    if (magic != MANIFEST_MAGIC) {
        LOG_WRN("checkpoint_db: bad manifest magic, ignoring\n");
        return;
    }

    const auto version = read_u32(is);
    if (version < 1 || version > MANIFEST_VERSION) {
        LOG_WRN("checkpoint_db: unsupported manifest version %u, ignoring\n", version);
        return;
    }

    // version 2+: read model_id, skip entirely if mismatched
    if (version >= 2) {
        const auto mid_len = read_u32(is);
        std::string manifest_mid;
        if (mid_len > 0) {
            manifest_mid.resize(mid_len);
            is.read(&manifest_mid[0], mid_len);
        }
        if (!model_id_.empty() && manifest_mid != model_id_) {
            LOG_WRN("checkpoint_db: model mismatch, discarding all %u entries\n", read_u32(is));
            return;
        }
    } else if (!model_id_.empty()) {
        // version 1 has no model_id — discard if we have a model identity
        LOG_WRN("checkpoint_db: discarding v1 manifest (no model_id, current model=%.32s)\n", model_id_.c_str());
        return;
    }

    const auto n = read_u32(is);
    LOG_INF("checkpoint_db: loading %u entries from manifest\n", n);

    for (uint32_t i = 0; i < n; ++i) {
        auto e = std::make_unique<entry>();
        e->id          = read_u32(is);
        const auto nt  = read_u64(is);
        e->tokens.resize(nt);
        is.read(reinterpret_cast<char*>(e->tokens.data()), nt * sizeof(llama_token));

        const auto kv_main_sz = read_u64(is);
        const auto kv_drft_sz = read_u64(is);
        // leave kv_main/kv_drft empty — find() loads them from disk on demand

        const auto nck = read_u32(is);
        for (uint32_t j = 0; j < nck; ++j) {
            common_prompt_checkpoint ckpt;
            ckpt.n_tokens = read_i64(is);
            ckpt.pos_min  = read_i32(is);
            ckpt.pos_max  = read_i32(is);
            auto ckpt_tgt_sz = read_u64(is);
            auto ckpt_dft_sz = read_u64(is);
            ckpt.data_tgt.resize(ckpt_tgt_sz);
            ckpt.data_dft.resize(ckpt_dft_sz);
            e->checkpoints.push_back(std::move(ckpt));
        }

        e->disk_offset = read_u64(is);
        e->disk_size   = read_u32(is);
        e->on_disk     = true;
        e->last_access = 0;

        total_disk_ += e->disk_size;
        entries_.push_back(std::move(e));
    }

    // rebuild trie
    for (auto & e : entries_) {
        insert_into_trie(e.get());
        lru_.push_back(e.get());
    }

    LOG_INF("checkpoint_db: loaded %zu entries (%.3f MiB on disk)\n",
            entries_.size(), total_disk_ / (1024.0 * 1024.0));
}

// --- storage ---

void checkpoint_db::store(
    const llama_tokens & tokens,
    const std::vector<uint8_t> & kv_main,
    const std::vector<uint8_t> & kv_drft,
    const std::list<common_prompt_checkpoint> & checkpoints)
{
    evict();

    auto * e = create_entry(tokens, kv_main, kv_drft, checkpoints);

    // always write to disk when disk path is configured (for crash recovery)
    if (!cfg_.disk_path.empty()) {
        write_entry_to_disk(e);
        // free RAM if over capacity
        if (cfg_.ram_capacity_mib > 0 && total_ram_ > cfg_.ram_capacity_mib * 1024ULL * 1024ULL) {
            e->kv_main.clear();
            e->kv_main.shrink_to_fit();
            e->kv_drft.clear();
            e->kv_drft.shrink_to_fit();
            e->checkpoints.clear();
        } else {
            total_ram_ += e->disk_size;
        }
    } else {
        total_ram_ += e->disk_size;
    }

    if (!cfg_.disk_path.empty()) {
        write_manifest();
    }
}

checkpoint_db::match_result checkpoint_db::find(const llama_tokens & query) {
    match_result res;
    if (!root_) {
        return res;
    }

    const entry * best = nullptr;
    size_t depth = 0;
    trie_node * node = root_.get();

    for (size_t i = 0; i < query.size(); ++i) {
        auto it = node->children.find(query[i]);
        if (it == node->children.end()) {
            break;
        }
        node = it->second.get();
        depth = i + 1;

        if (node->ent) {
            best = node->ent;
            res.n_matched = depth;
        }
    }

    if (!best) {
        return res;
    }

    // load from disk if needed
    entry * e = const_cast<entry*>(best);
    touch_lru(e);
    if (e->on_disk && e->kv_main.empty()) {
        read_entry_from_disk(e);
        e->on_disk = false;
        total_disk_ -= e->disk_size;
        total_ram_  += e->disk_size;
    }

    res.found = true;
    res.matched_tokens = e->tokens;
    res.kv_main       = e->kv_main;
    res.kv_drft       = e->kv_drft;
    res.checkpoints   = e->checkpoints;

    return res;
}

void checkpoint_db::evict() {
    if (cfg_.disk_path.empty()) {
        // RAM-only mode: evict LRU when over ram_capacity
        while (total_ram_ > cfg_.ram_capacity_mib * 1024ULL * 1024ULL && !lru_.empty()) {
            evict_one();
        }
        return;
    }

    // disk mode: first ensure all entries are on disk
    // then evict oldest from RAM when over capacity
    for (auto & e : entries_) {
        if (!e->on_disk && !e->kv_main.empty()) {
            write_entry_to_disk(e.get());
        }
    }

    // also bound disk usage
    while (total_disk_ > cfg_.disk_capacity_mib * 1024ULL * 1024ULL && !lru_.empty()) {
        remove_entry_from_disk(lru_.back());
    }

    if (!cfg_.disk_path.empty()) {
        write_manifest();
    }
}

void checkpoint_db::flush() {
    if (cfg_.disk_path.empty()) {
        return;
    }

    fs::create_directories(cfg_.disk_path + "/e");

    LOG_INF("checkpoint_db: flushing %zu entries to disk\n", entries_.size());

    for (auto & e : entries_) {
        write_entry_to_disk(e.get());
    }

    write_manifest();
}

void checkpoint_db::write_manifest() {
    const auto mpath = manifest_path();
    std::ofstream os(mpath, std::ios::binary);
    if (!os.is_open()) {
        LOG_ERR("checkpoint_db: cannot write manifest to %s\n", mpath.c_str());
        return;
    }

    write_u32(os, MANIFEST_MAGIC);
    write_u32(os, MANIFEST_VERSION);
    // version 2: write model_id for cross-model safety
    write_u32(os, static_cast<uint32_t>(model_id_.size()));
    if (!model_id_.empty()) {
        os.write(model_id_.data(), model_id_.size());
    }
    write_u32(os, static_cast<uint32_t>(entries_.size()));

    for (const auto & e : entries_) {
        write_u32(os, e->id);
        write_u64(os, e->tokens.size());
        os.write(reinterpret_cast<const char*>(e->tokens.data()), e->tokens.size() * sizeof(llama_token));

        write_u64(os, e->kv_main.size());
        write_u64(os, e->kv_drft.size());

        write_u32(os, static_cast<uint32_t>(e->checkpoints.size()));
        for (const auto & ckpt : e->checkpoints) {
            write_i64(os, ckpt.n_tokens);
            write_i32(os, ckpt.pos_min);
            write_i32(os, ckpt.pos_max);
            write_u64(os, ckpt.data_tgt.size());
            write_u64(os, ckpt.data_dft.size());
        }

        write_u64(os, e->disk_offset);
        write_u32(os, e->disk_size);
    }

    LOG_INF("checkpoint_db: wrote manifest with %zu entries\n", entries_.size());
}

void checkpoint_db::read_manifest() {
    // alias for load_manifest
    load_manifest();
}

// --- private helpers ---

checkpoint_db::entry * checkpoint_db::create_entry(
    const llama_tokens & tokens,
    const std::vector<uint8_t> & kv_main,
    const std::vector<uint8_t> & kv_drft,
    const std::list<common_prompt_checkpoint> & checkpoints)
{
    auto e = std::make_unique<entry>();
    e->id     = next_id_++;
    e->tokens = tokens;
    e->kv_main = kv_main;
    e->kv_drft = kv_drft;
    e->checkpoints = checkpoints;
    e->on_disk     = false;
    e->disk_offset = 0;
    e->disk_size   = 0;
    e->last_access = ggml_time_us();

    // estimate disk size
    e->disk_size = static_cast<uint32_t>(kv_main.size() + kv_drft.size());
    for (const auto & ckpt : checkpoints) {
        e->disk_size += static_cast<uint32_t>(ckpt.data_tgt.size() + ckpt.data_dft.size());
    }

    entry * ptr = e.get();
    entries_.push_back(std::move(e));
    insert_into_trie(ptr);
    lru_.push_back(ptr);
    touch_lru(ptr);

    return ptr;
}

void checkpoint_db::insert_into_trie(entry * e) {
    trie_node * node = root_.get();
    for (size_t i = 0; i < e->tokens.size(); ++i) {
        const auto tok = e->tokens[i];
        if (!node->children[tok]) {
            node->children[tok] = std::make_unique<trie_node>();
        }
        node = node->children[tok].get();
        // set entry at every prefix node so shorter queries find matches too
        node->ent = e;
    }
}

void checkpoint_db::touch_lru(entry * e) {
    e->last_access = ggml_time_us();

    // bubble to front in lru list
    auto it = std::find(lru_.begin(), lru_.end(), e);
    if (it != lru_.end() && it != lru_.begin()) {
        lru_.splice(lru_.begin(), lru_, it);
    }
}

void checkpoint_db::evict_one() {
    if (lru_.empty()) {
        return;
    }

    entry * e = lru_.back();

    // free RAM
    total_ram_ -= e->disk_size;

    if (!cfg_.disk_path.empty()) {
        write_entry_to_disk(e);
    }

    e->kv_main.clear();
    e->kv_main.shrink_to_fit();
    e->kv_drft.clear();
    e->kv_drft.shrink_to_fit();
    e->checkpoints.clear();
    e->on_disk = true;

    lru_.pop_back();
    lru_.push_front(e);
}

void checkpoint_db::write_entry_to_disk(entry * e) {
    if (cfg_.disk_path.empty() || !e) {
        return;
    }

    fs::create_directories(cfg_.disk_path + "/e");

    const auto kpath = kv_path(e->id);
    std::ofstream os(kpath, std::ios::binary);
    if (!os.is_open()) {
        LOG_ERR("checkpoint_db: cannot write %s\n", kpath.c_str());
        return;
    }

    // format:
    //   uint32_t kv_main_size
    //   uint8_t[kv_main_size]
    //   uint32_t kv_drft_size
    //   uint8_t[kv_drft_size]
    //   uint32_t n_checkpoints
    //   for each:
    //     int64_t  n_tokens
    //     int32_t  pos_min
    //     int32_t  pos_max
    //     uint32_t data_tgt_size
    //     uint8_t[data_tgt_size]
    //     uint32_t data_dft_size
    //     uint8_t[data_dft_size]

    const auto kv_main_sz = static_cast<uint32_t>(e->kv_main.size());
    const auto kv_drft_sz = static_cast<uint32_t>(e->kv_drft.size());

    write_u32(os, kv_main_sz);
    if (kv_main_sz > 0) {
        os.write(reinterpret_cast<const char*>(e->kv_main.data()), kv_main_sz);
    }

    write_u32(os, kv_drft_sz);
    if (kv_drft_sz > 0) {
        os.write(reinterpret_cast<const char*>(e->kv_drft.data()), kv_drft_sz);
    }

    write_u32(os, static_cast<uint32_t>(e->checkpoints.size()));
    for (const auto & ckpt : e->checkpoints) {
        write_i64(os, ckpt.n_tokens);
        write_i32(os, ckpt.pos_min);
        write_i32(os, ckpt.pos_max);

        const auto tgt_sz = static_cast<uint32_t>(ckpt.data_tgt.size());
        write_u32(os, tgt_sz);
        if (tgt_sz > 0) {
            os.write(reinterpret_cast<const char*>(ckpt.data_tgt.data()), tgt_sz);
        }

        const auto dft_sz = static_cast<uint32_t>(ckpt.data_dft.size());
        write_u32(os, dft_sz);
        if (dft_sz > 0) {
            os.write(reinterpret_cast<const char*>(ckpt.data_dft.data()), dft_sz);
        }
    }

    const auto file_size = static_cast<uint32_t>(os.tellp());
    if (e->on_disk) {
        // update disk size (may have changed)
        total_disk_ -= e->disk_size;
    }
    total_disk_ += file_size;

    e->disk_offset = 0;
    e->disk_size   = file_size;
    e->on_disk     = true;

    LOG_DBG("checkpoint_db: wrote entry %u to %s (%u bytes)\n", e->id, kpath.c_str(), file_size);
}

void checkpoint_db::read_entry_from_disk(entry * e) const {
    if (!e || !e->on_disk) {
        return;
    }

    const auto kpath = kv_path(e->id);
    std::ifstream is(kpath, std::ios::binary);
    if (!is.is_open()) {
        LOG_ERR("checkpoint_db: cannot read %s\n", kpath.c_str());
        return;
    }

    const auto kv_main_sz = read_u32(is);
    e->kv_main.resize(kv_main_sz);
    if (kv_main_sz > 0) {
        is.read(reinterpret_cast<char*>(e->kv_main.data()), kv_main_sz);
    }

    const auto kv_drft_sz = read_u32(is);
    e->kv_drft.resize(kv_drft_sz);
    if (kv_drft_sz > 0) {
        is.read(reinterpret_cast<char*>(e->kv_drft.data()), kv_drft_sz);
    }

    const auto nck = read_u32(is);
    e->checkpoints.clear();
    for (uint32_t i = 0; i < nck; ++i) {
        common_prompt_checkpoint ckpt;
        ckpt.n_tokens = read_i64(is);
        ckpt.pos_min  = read_i32(is);
        ckpt.pos_max  = read_i32(is);

        auto tgt_sz = read_u32(is);
        ckpt.data_tgt.resize(tgt_sz);
        if (tgt_sz > 0) {
            is.read(reinterpret_cast<char*>(ckpt.data_tgt.data()), tgt_sz);
        }

        auto dft_sz = read_u32(is);
        ckpt.data_dft.resize(dft_sz);
        if (dft_sz > 0) {
            is.read(reinterpret_cast<char*>(ckpt.data_dft.data()), dft_sz);
        }

        e->checkpoints.push_back(std::move(ckpt));
    }

    e->on_disk = false;

    LOG_DBG("checkpoint_db: read entry %u from %s\n", e->id, kpath.c_str());
}

void checkpoint_db::remove_entry_from_disk(entry * e) {
    // remove from LRU and trie
    // first, remove trie linkage
    if (root_) {
        trie_node * node = root_.get();
        for (size_t i = 0; i < e->tokens.size(); ++i) {
            auto it = node->children.find(e->tokens[i]);
            if (it == node->children.end()) {
                break;
            }
            node = it->second.get();
        }
        if (node->ent == e) {
            node->ent = nullptr;
        }
    }

    // remove disk file
    if (e->on_disk) {
        const auto kpath = kv_path(e->id);
        std::error_code ec;
        fs::remove(kpath, ec);
        total_disk_ -= e->disk_size;
    }

    // remove from LRU
    auto it = std::find(lru_.begin(), lru_.end(), e);
    if (it != lru_.end()) {
        lru_.erase(it);
    }

    // remove from entries vector
    auto eit = std::find_if(entries_.begin(), entries_.end(),
        [e](const auto & p) { return p.get() == e; });
    if (eit != entries_.end()) {
        total_ram_ -= e->disk_size;
        entries_.erase(eit);
    }
}
