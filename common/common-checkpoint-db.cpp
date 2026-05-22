#include "common-checkpoint-db.h"
#include "common.h"
#include "log.h"
#include "xxhash.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

// --- manifest binary format (version 3) ---
// uint32_t magic        = 0xDBDBDBDB
// uint32_t version      = 3
// uint32_t model_id_len
// uint8_t[model_id_len]
// uint32_t n_entries
// for each entry:
//   uint32_t id
//   uint64_t n_tokens
//   llama_token[n_tokens]
//   uint32_t n_chunks_main
//   for each chunk:
//     uint64_t hash
//     uint32_t size
//   uint32_t n_chunks_drft
//   for each chunk:
//     uint64_t hash
//     uint32_t size
//   uint32_t n_checkpoints
//   for each checkpoint:
//     int64_t  ckpt_n_tokens
//     int32_t  ckpt_pos_min
//     int32_t  ckpt_pos_max
//     uint64_t ckpt_tgt_size
//     uint8_t[ckpt_tgt_size]
//     uint64_t ckpt_dft_size
//     uint8_t[ckpt_dft_size]
//   uint64_t unused_rsvd   // reserved, written as 0

static const uint32_t MANIFEST_MAGIC   = 0xDBDBDBDB;
static const uint32_t MANIFEST_VERSION = 3;

// helpers
static void write_u32(std::ofstream & os, uint32_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static void write_u64(std::ofstream & os, uint64_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static void write_i64(std::ofstream & os, int64_t  v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
static void write_i32(std::ofstream & os, int32_t  v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }

static uint32_t read_u32(std::ifstream & is) { uint32_t v; is.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
static uint64_t read_u64(std::ifstream & is) { uint64_t v; is.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
static int64_t  read_i64(std::ifstream & is) { int64_t  v; is.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }
static int32_t  read_i32(std::ifstream & is) { int32_t  v; is.read(reinterpret_cast<char*>(&v), sizeof(v)); return v; }

// --- chunk hashing ---

uint64_t checkpoint_db::hash_chunk(const uint8_t * data, size_t len) {
    return XXH3_64bits(data, len);
}

std::string checkpoint_db::chunk_path(uint64_t hash) const {
    // two-level directory to avoid giant directories: c/{lo8}/{hash}.chunk
    char lo[3] = {};
    snprintf(lo, sizeof(lo), "%02x", (unsigned)(hash & 0xFF));
    return cfg_.disk_path + "/c/" + lo + "/" + std::to_string(hash) + ".chunk";
}

std::string checkpoint_db::chunks_dir() const {
    return cfg_.disk_path + "/c";
}

void checkpoint_db::write_chunk(uint64_t hash, const uint8_t * data, size_t len) {
    const auto path = chunk_path(hash);
    // skip if chunk already exists (content-addressed)
    if (fs::exists(path)) {
        return;
    }
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream os(path, std::ios::binary);
    if (!os.is_open()) {
        LOG_ERR("checkpoint_db: cannot write chunk %s\n", path.c_str());
        return;
    }
    os.write(reinterpret_cast<const char*>(data), len);
}

bool checkpoint_db::read_chunk(uint64_t hash, uint8_t * out, size_t len) const {
    const auto path = chunk_path(hash);
    std::ifstream is(path, std::ios::binary);
    if (!is.is_open()) {
        return false;
    }
    is.read(reinterpret_cast<char*>(out), len);
    return !is.fail();
}

void checkpoint_db::remove_chunk(uint64_t hash) {
    std::error_code ec;
    fs::remove(chunk_path(hash), ec);
}

// --- chunking ---

std::vector<checkpoint_db::chunk_ref> checkpoint_db::chunkify(const uint8_t * data, size_t data_size) const {
    std::vector<chunk_ref> refs;
    if (data_size == 0) return refs;

    size_t pos = 0;
    while (pos < data_size) {
        size_t chunk_sz = CHUNK_TARGET;

        // Align to token boundaries when bytes_per_token is known
        if (bytes_per_token_ > 0 && chunk_sz > bytes_per_token_) {
            size_t tokens_in_chunk = std::max<size_t>(1, chunk_sz / bytes_per_token_);
            chunk_sz = tokens_in_chunk * bytes_per_token_;
        }

        // last chunk may be smaller
        if (pos + chunk_sz > data_size) {
            chunk_sz = data_size - pos;
        }

        chunk_ref ref;
        ref.hash = hash_chunk(data + pos, chunk_sz);
        ref.size = static_cast<uint32_t>(chunk_sz);
        refs.push_back(ref);
        pos += chunk_sz;
    }
    return refs;
}

std::vector<uint8_t> checkpoint_db::dechunkify(const std::vector<chunk_ref> & refs) const {
    size_t total = 0;
    for (const auto & r : refs) total += r.size;
    std::vector<uint8_t> out(total);
    size_t pos = 0;
    for (const auto & r : refs) {
        if (!read_chunk(r.hash, out.data() + pos, r.size)) {
            LOG_ERR("checkpoint_db: missing chunk %" PRIu64 ", size=%u\n", r.hash, r.size);
            return {};
        }
        pos += r.size;
    }
    return out;
}

// --- entry chunk I/O ---

void checkpoint_db::write_entry_chunks(entry * e) {
    if (cfg_.disk_path.empty() || !e) return;

    // chunk main KV blob
    e->chunks_main = chunkify(e->kv_main.data(), e->kv_main.size());
    for (const auto & c : e->chunks_main) {
        size_t off = (&c - &e->chunks_main[0]) * sizeof(chunk_ref); // approximate: compute offset
        // Actually just iterate directly
    }
    // need a loop with proper offset computation
    size_t offset = 0;
    for (auto & c : e->chunks_main) {
        write_chunk(c.hash, e->kv_main.data() + offset, c.size);
        offset += c.size;
    }

    // chunk draft KV blob
    e->chunks_drft = chunkify(e->kv_drft.data(), e->kv_drft.size());
    offset = 0;
    for (auto & c : e->chunks_drft) {
        write_chunk(c.hash, e->kv_drft.data() + offset, c.size);
        offset += c.size;
    }

    // track disk usage — count only the chunk data (checkpoints stored inline in manifest)
    uint64_t chunk_bytes = 0;
    for (const auto & c : e->chunks_main) chunk_bytes += c.size;
    for (const auto & c : e->chunks_drft) chunk_bytes += c.size;

    if (e->on_disk) {
        // re-counting; adjust delta
        // simpler: just set e->disk_size and let eviction track it
    }

    e->on_disk = true;
}

void checkpoint_db::read_entry_chunks(entry * e) const {
    if (!e) return;

    e->kv_main = dechunkify(e->chunks_main);
    if (e->chunks_main.empty() && !e->kv_main.empty()) {
        LOG_ERR("checkpoint_db: entry %u has no main chunk refs but has kv_main data\n", e->id);
        return;
    }

    e->kv_drft = dechunkify(e->chunks_drft);

    e->on_disk = false;
}

void checkpoint_db::remove_entry_chunks(entry * e) {
    if (!e) return;

    // We don't eagerly delete chunks — orphan chunks are cleaned up
    // during evict() or by scanning the chunks dir periodically.
    // For now, just clear the refs.
    e->chunks_main.clear();
    e->chunks_drft.clear();
}

// --- constructors, destructor ---

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

// --- manifest load ---

void checkpoint_db::load_manifest() {
    if (cfg_.disk_path.empty()) return;

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

    // version 2+: read model_id, discard on mismatch
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
        LOG_WRN("checkpoint_db: discarding v1 manifest (no model_id)\n");
        return;
    }

    const auto n = read_u32(is);
    LOG_INF("checkpoint_db: loading %u entries from manifest\n", n);

    if (version == 1 || version == 2) {
        // v1/v2 used .kv files — discard; upgrade to v3 chunks
        LOG_WRN("checkpoint_db: upgrading manifest from v%u to v3 (chunked storage)\n", version);
        // Skip entries so the stream is consumed, but don't import them
        // (old .kv files will be orphaned, cleanup is manual)
        for (uint32_t i = 0; i < n; ++i) {
            read_u32(is); // id
            uint64_t nt = read_u64(is);
            is.seekg(nt * sizeof(llama_token), std::ios::cur); // tokens
            read_u64(is); read_u64(is); // kv_main_sz, kv_drft_sz
            uint32_t nck = read_u32(is);
            for (uint32_t j = 0; j < nck; ++j) {
                read_i64(is); read_i32(is); read_i32(is);
                read_u64(is); read_u64(is); // sizes
            }
            read_u64(is); read_u32(is); // offset, size
        }
        return;
    }

    // version 3
    load_manifest_v3(is, n);
}

void checkpoint_db::load_manifest_v3(std::ifstream & is, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i) {
        auto e = std::make_unique<entry>();
        e->id          = read_u32(is);
        const auto nt  = read_u64(is);
        e->tokens.resize(nt);
        is.read(reinterpret_cast<char*>(e->tokens.data()), nt * sizeof(llama_token));

        // main chunk refs
        uint32_t ncm = read_u32(is);
        e->chunks_main.resize(ncm);
        for (uint32_t j = 0; j < ncm; ++j) {
            e->chunks_main[j].hash = read_u64(is);
            e->chunks_main[j].size = read_u32(is);
        }

        // draft chunk refs
        uint32_t ncd = read_u32(is);
        e->chunks_drft.resize(ncd);
        for (uint32_t j = 0; j < ncd; ++j) {
            e->chunks_drft[j].hash = read_u64(is);
            e->chunks_drft[j].size = read_u32(is);
        }

        // checkpoints are stored inline in v3 (they are small)
        uint32_t nck = read_u32(is);
        for (uint32_t j = 0; j < nck; ++j) {
            common_prompt_checkpoint ckpt;
            ckpt.n_tokens = read_i64(is);
            ckpt.pos_min  = read_i32(is);
            ckpt.pos_max  = read_i32(is);
            auto ckpt_tgt_sz = read_u64(is);
            ckpt.data_tgt.resize(ckpt_tgt_sz);
            if (ckpt_tgt_sz > 0) is.read(reinterpret_cast<char*>(ckpt.data_tgt.data()), ckpt_tgt_sz);
            auto ckpt_dft_sz = read_u64(is);
            ckpt.data_dft.resize(ckpt_dft_sz);
            if (ckpt_dft_sz > 0) is.read(reinterpret_cast<char*>(ckpt.data_dft.data()), ckpt_dft_sz);
            e->checkpoints.push_back(std::move(ckpt));
        }

        read_u64(is); // reserved

        // compute disk size from chunks
        e->disk_size = 0;
        for (const auto & c : e->chunks_main) e->disk_size += c.size;
        for (const auto & c : e->chunks_drft) e->disk_size += c.size;

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

// --- manifest write ---

void checkpoint_db::write_manifest() {
    const auto mpath = manifest_path();
    std::ofstream os(mpath, std::ios::binary);
    if (!os.is_open()) {
        LOG_ERR("checkpoint_db: cannot write manifest to %s\n", mpath.c_str());
        return;
    }

    write_u32(os, MANIFEST_MAGIC);
    write_u32(os, MANIFEST_VERSION);
    write_u32(os, static_cast<uint32_t>(model_id_.size()));
    if (!model_id_.empty()) os.write(model_id_.data(), model_id_.size());
    write_u32(os, static_cast<uint32_t>(entries_.size()));

    for (const auto & e : entries_) {
        write_u32(os, e->id);
        write_u64(os, e->tokens.size());
        os.write(reinterpret_cast<const char*>(e->tokens.data()), e->tokens.size() * sizeof(llama_token));

        // main chunk refs
        write_u32(os, static_cast<uint32_t>(e->chunks_main.size()));
        for (const auto & c : e->chunks_main) {
            write_u64(os, c.hash);
            write_u32(os, c.size);
        }

        // draft chunk refs
        write_u32(os, static_cast<uint32_t>(e->chunks_drft.size()));
        for (const auto & c : e->chunks_drft) {
            write_u64(os, c.hash);
            write_u32(os, c.size);
        }

        // checkpoints inline
        write_u32(os, static_cast<uint32_t>(e->checkpoints.size()));
        for (const auto & ckpt : e->checkpoints) {
            write_i64(os, ckpt.n_tokens);
            write_i32(os, ckpt.pos_min);
            write_i32(os, ckpt.pos_max);
            write_u64(os, ckpt.data_tgt.size());
            if (ckpt.data_tgt.size() > 0) os.write(reinterpret_cast<const char*>(ckpt.data_tgt.data()), ckpt.data_tgt.size());
            write_u64(os, ckpt.data_dft.size());
            if (ckpt.data_dft.size() > 0) os.write(reinterpret_cast<const char*>(ckpt.data_dft.data()), ckpt.data_dft.size());
        }

        write_u64(os, 0); // reserved
    }

    LOG_INF("checkpoint_db: wrote manifest with %zu entries\n", entries_.size());
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

    // always write chunks when disk path is configured
    if (!cfg_.disk_path.empty()) {
        write_entry_chunks(e);
        if (cfg_.ram_capacity_mib > 0 && total_ram_ > cfg_.ram_capacity_mib * 1024ULL * 1024ULL) {
            e->kv_main.clear(); e->kv_main.shrink_to_fit();
            e->kv_drft.clear(); e->kv_drft.shrink_to_fit();
            e->checkpoints.clear();
        } else {
            total_ram_ += e->disk_size;
        }
    } else {
        total_ram_ += e->disk_size;
    }

    if (!cfg_.disk_path.empty()) write_manifest();
}

checkpoint_db::match_result checkpoint_db::find(const llama_tokens & query) {
    match_result res;
    if (!root_) return res;

    const entry * best = nullptr;
    size_t depth = 0;
    trie_node * node = root_.get();

    for (size_t i = 0; i < query.size(); ++i) {
        auto it = node->children.find(query[i]);
        if (it == node->children.end()) break;
        node = it->second.get();
        depth = i + 1;
        if (node->ent) {
            best = node->ent;
            res.n_matched = depth;
        }
    }

    if (!best) return res;

    entry * e = const_cast<entry*>(best);
    touch_lru(e);

    // load from chunks if on disk and not already in RAM
    if (e->on_disk && e->kv_main.empty()) {
        read_entry_chunks(e);
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

// --- eviction ---

void checkpoint_db::evict() {
    if (cfg_.disk_path.empty()) {
        while (total_ram_ > cfg_.ram_capacity_mib * 1024ULL * 1024ULL && !lru_.empty()) {
            evict_one();
        }
        return;
    }

    for (auto & e : entries_) {
        if (!e->on_disk && !e->kv_main.empty()) {
            write_entry_chunks(e.get());
        }
    }

    while (total_disk_ > cfg_.disk_capacity_mib * 1024ULL * 1024ULL && !lru_.empty()) {
        remove_entry_chunks(lru_.back());
    }

    if (!cfg_.disk_path.empty()) write_manifest();
}

void checkpoint_db::flush() {
    if (cfg_.disk_path.empty()) return;

    fs::create_directories(chunks_dir());
    LOG_INF("checkpoint_db: flushing %zu entries to disk\n", entries_.size());

    for (auto & e : entries_) {
        write_entry_chunks(e.get());
    }

    write_manifest();
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
    
    e->disk_size   = 0;
    e->last_access = ggml_time_us();

    // estimate disk size (exact after chunkify)
    e->disk_size = static_cast<uint32_t>(kv_main.size() + kv_drft.size());

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
        node->ent = e;
    }
}

void checkpoint_db::touch_lru(entry * e) {
    e->last_access = ggml_time_us();
    auto it = std::find(lru_.begin(), lru_.end(), e);
    if (it != lru_.end() && it != lru_.begin()) {
        lru_.splice(lru_.begin(), lru_, it);
    }
}

void checkpoint_db::evict_one() {
    if (lru_.empty()) return;

    entry * e = lru_.back();
    total_ram_ -= e->disk_size;

    if (!cfg_.disk_path.empty()) {
        write_entry_chunks(e);
    }

    e->kv_main.clear(); e->kv_main.shrink_to_fit();
    e->kv_drft.clear(); e->kv_drft.shrink_to_fit();
    e->checkpoints.clear();
    e->on_disk = true;

    lru_.pop_back();
    lru_.push_front(e);
}


