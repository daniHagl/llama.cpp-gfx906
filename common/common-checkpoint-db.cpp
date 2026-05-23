#include "common-checkpoint-db.h"
#include "common.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

// --- manifest format v2 (current) and v3 (with cell refs) ---
// v2: used for .kv file storage
// v3: used for cell-based dedup storage
// Both share magic + model_id + entry structure, v3 adds cell refs per entry.
//
// v3 entry format (difference from v2):
//   uint32_t id
//   uint64_t n_tokens
//   llama_token[n_tokens]
//   uint64_t kv_main_header_size
//   uint8_t[kv_main_header_size]       // header bytes (before cell data)
//   uint32_t n_cells_main
//   for each cell:
//     uint64_t hash
//     uint32_t size
//   uint64_t kv_drft_header_size      // same for draft; 0 if no draft
//   uint8_t[kv_drft_header_size]
//   uint32_t n_cells_drft
//   for each cell:
//     uint64_t hash
//     uint32_t size
//   uint32_t n_checkpoints
//   for each checkpoint:
//     (same as v2)
//   uint64_t reserved (0)

static const uint32_t MANIFEST_MAGIC   = 0xDBDBDBDB;
static const uint32_t MANIFEST_VERSION = 3;

// helpers
static void w_u32(std::ofstream & os, uint32_t v) { os.write((const char*)&v, sizeof(v)); }
static void w_u64(std::ofstream & os, uint64_t v) { os.write((const char*)&v, sizeof(v)); }
static void w_i64(std::ofstream & os, int64_t  v) { os.write((const char*)&v, sizeof(v)); }
static void w_i32(std::ofstream & os, int32_t  v) { os.write((const char*)&v, sizeof(v)); }

static uint32_t r_u32(std::ifstream & is) { uint32_t v; is.read((char*)&v, sizeof(v)); return v; }
static uint64_t r_u64(std::ifstream & is) { uint64_t v; is.read((char*)&v, sizeof(v)); return v; }
static int64_t  r_i64(std::ifstream & is) { int64_t  v; is.read((char*)&v, sizeof(v)); return v; }
static int32_t  r_i32(std::ifstream & is) { int32_t  v; is.read((char*)&v, sizeof(v)); return v; }

// --- FNV-1a hashing ---

uint64_t checkpoint_db::hash_bytes(const uint8_t * data, size_t len) {
    uint64_t h = 0xCBF29CE484222325ULL;
    for (size_t i = 0; i < len; ++i) { h ^= data[i]; h *= 0x100000001B3ULL; }
    return h;
}

std::string checkpoint_db::cell_path(uint64_t hash) const {
    char buf[3] = {}; snprintf(buf, sizeof(buf), "%02x", (unsigned)(hash & 0xFF));
    return cfg_.disk_path + "/d/" + buf + "/" + std::to_string(hash) + ".cell";
}

void checkpoint_db::write_cell(uint64_t hash, const uint8_t * data, size_t len) {
    auto path = cell_path(hash);
    if (fs::exists(path)) return; // content-addressed: already stored
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream os(path, std::ios::binary);
    if (!os.is_open()) { LOG_ERR("checkpoint_db: cannot write cell %s\n", path.c_str()); return; }
    os.write((const char*)data, len);
}

bool checkpoint_db::read_cell(uint64_t hash, uint8_t * out, size_t len) const {
    auto path = cell_path(hash);
    std::ifstream is(path, std::ios::binary);
    if (!is.is_open()) return false;
    is.read((char*)out, len);
    return !is.fail();
}

void checkpoint_db::remove_cell(uint64_t hash) {
    std::error_code ec; fs::remove(cell_path(hash), ec);
}

// --- Reconstruct full KV blob from header + cell data ---

std::vector<uint8_t> checkpoint_db::reconstruct_blob(
    const std::vector<uint8_t> & header,
    const std::vector<cell_ref> & cells,
    const uint8_t * cell_data_buf)
{
    if (header.empty() || cells.empty()) return {};

    // Parse header to get n_layer and per-layer sizes
    const uint8_t * p = header.data();
    const uint8_t * end = p + header.size();

    if (p + 4 > end) return {};
    uint32_t n_stream = 0; memcpy(&n_stream, p, 4); p += 4;
    if (n_stream != 1) return {}; // only support n_stream=1

    if (p + 4 > end) return {};
    uint32_t cell_count = 0; memcpy(&cell_count, p, 4); p += 4;

    // Skip per-cell metadata
    for (uint32_t i = 0; i < cell_count && p + 8 <= end; ++i) {
        p += 4; // pos
        uint32_t ns = 0; memcpy(&ns, p, 4); p += 4;
        p += ns * 4; // seq_ids
    }

    if (p + 8 > end) return {};
    uint32_t v_trans = 0; memcpy(&v_trans, p, 4); p += 4;
    uint32_t n_layer = 0; memcpy(&n_layer, p, 4); p += 4;

    struct li { uint64_t kr; uint64_t vr; };
    std::vector<li> layers;
    for (uint32_t l = 0; l < n_layer && p + 12 <= end; ++l) {
        p += 4; // k_type_i
        uint64_t kr = 0; memcpy(&kr, p, 8); p += 8;
        if (!v_trans) {
            p += 4; // v_type_i
            uint64_t vr = 0; memcpy(&vr, p, 8); p += 8;
            layers.push_back({kr, vr});
        } else {
            p += 4; // v_type_i
            uint32_t ve = 0; memcpy(&ve, p, 4); p += 4;
            uint32_t nv = 0; memcpy(&nv, p, 4); p += 4;
            layers.push_back({kr, (uint64_t)nv * ve});
        }
    }

    // Build blob: header + layer-major cell data
    std::vector<uint8_t> blob;
    blob.insert(blob.end(), header.begin(), header.end());

    const uint8_t * cd = cell_data_buf;
    for (const auto & li : layers) {
        for (size_t i = 0; i < cells.size(); ++i) {
            blob.insert(blob.end(), cd, cd + li.kr);
            cd += li.kr;
        }
        for (size_t i = 0; i < cells.size(); ++i) {
            blob.insert(blob.end(), cd, cd + li.vr);
            cd += li.vr;
        }
    }
    return blob;
}

// --- constructor / destructor ---

checkpoint_db::checkpoint_db(const checkpoint_db_config & cfg)
    : cfg_(cfg), root_(std::make_unique<trie_node>()) {}

checkpoint_db::~checkpoint_db() { try { flush(); } catch (...) {} }

std::string checkpoint_db::manifest_path() const {
    return cfg_.disk_path + "/manifest.bin";
}

std::string checkpoint_db::kv_path(uint32_t id) const {
    return cfg_.disk_path + "/e/" + std::to_string(id) + ".kv";
}

// --- manifest load ---

void checkpoint_db::load_manifest() {
    if (cfg_.disk_path.empty()) return;
    auto mpath = manifest_path();
    std::ifstream is(mpath, std::ios::binary);
    if (!is.is_open()) { LOG_INF("checkpoint_db: no manifest at %s\n", mpath.c_str()); return; }

    auto magic = r_u32(is);
    if (magic != MANIFEST_MAGIC) { LOG_WRN("checkpoint_db: bad magic\n"); return; }
    auto ver = r_u32(is);
    if (ver < 2 || ver > MANIFEST_VERSION) { LOG_WRN("checkpoint_db: unsupported v%u\n", ver); return; }

    auto mid_len = r_u32(is);
    std::string mmid;
    if (mid_len > 0) { mmid.resize(mid_len); is.read(&mmid[0], mid_len); }
    if (!model_id_.empty() && mmid != model_id_) { LOG_WRN("checkpoint_db: model mismatch\n"); return; }

    auto n = r_u32(is);
    LOG_INF("checkpoint_db: loading %u entries\n", n);

    if (ver == 2) {
        // v2: .kv file storage — skip entries to advance stream
        for (uint32_t i = 0; i < n; ++i) {
            r_u32(is); uint64_t nt = r_u64(is); is.seekg(nt * 4, std::ios::cur);
            r_u64(is); r_u64(is); // kv_main_sz, kv_drft_sz
            uint32_t nck = r_u32(is);
            for (uint32_t j = 0; j < nck; ++j) { r_i64(is); r_i32(is); r_i32(is); r_u64(is); r_u64(is); }
            r_u64(is); r_u32(is); // disk_offset, disk_size
        }
        LOG_WRN("checkpoint_db: discarding v2 entries (upgrade to v3)\n");
        return;
    }

    // v3: cell-based storage
    for (uint32_t i = 0; i < n; ++i) {
        auto e = std::make_unique<entry>();
        e->id = r_u32(is);
        uint64_t nt = r_u64(is);
        e->tokens.resize(nt);
        is.read((char*)e->tokens.data(), nt * 4);

        uint64_t hsz = r_u64(is);
        e->kv_main_header.resize(hsz);
        if (hsz > 0) is.read((char*)e->kv_main_header.data(), hsz);

        uint32_t ncm = r_u32(is);
        e->cells_main.resize(ncm);
        for (uint32_t j = 0; j < ncm; ++j) {
            e->cells_main[j].hash = r_u64(is);
            e->cells_main[j].size = r_u32(is);
        }

        uint64_t dhsz = r_u64(is);
        e->kv_drft_header.resize(dhsz);
        if (dhsz > 0) is.read((char*)e->kv_drft_header.data(), dhsz);

        uint32_t ncd = r_u32(is);
        e->cells_drft.resize(ncd);
        for (uint32_t j = 0; j < ncd; ++j) {
            e->cells_drft[j].hash = r_u64(is);
            e->cells_drft[j].size = r_u32(is);
        }

        uint32_t nck = r_u32(is);
        for (uint32_t j = 0; j < nck; ++j) {
            common_prompt_checkpoint ck;
            ck.n_tokens = r_i64(is); ck.pos_min = r_i32(is); ck.pos_max = r_i32(is);
            uint64_t ts = r_u64(is); ck.data_tgt.resize(ts); if (ts > 0) is.read((char*)ck.data_tgt.data(), ts);
            uint64_t ds = r_u64(is); ck.data_dft.resize(ds); if (ds > 0) is.read((char*)ck.data_dft.data(), ds);
            e->checkpoints.push_back(std::move(ck));
        }
        r_u64(is); // reserved

        // disk size = all cell data bytes
        e->disk_size = 0;
        for (const auto & c : e->cells_main) e->disk_size += c.size;
        for (const auto & c : e->cells_drft) e->disk_size += c.size;
        e->on_disk = true;
        e->last_access = 0;

        total_disk_ += e->disk_size;
        entries_.push_back(std::move(e));
    }

    for (auto & e : entries_) { insert_into_trie(e.get()); lru_.push_back(e.get()); }
    LOG_INF("checkpoint_db: loaded %zu entries (%.3f MiB)\n", entries_.size(), total_disk_ / (1024.0*1024.0));
}

// --- manifest write (v3) ---

void checkpoint_db::write_manifest() {
    auto mpath = manifest_path();
    std::ofstream os(mpath, std::ios::binary);
    if (!os.is_open()) { LOG_ERR("checkpoint_db: cannot write manifest\n"); return; }

    w_u32(os, MANIFEST_MAGIC);
    w_u32(os, MANIFEST_VERSION);
    w_u32(os, (uint32_t)model_id_.size());
    if (!model_id_.empty()) os.write(model_id_.data(), model_id_.size());
    w_u32(os, (uint32_t)entries_.size());

    for (const auto & e : entries_) {
        w_u32(os, e->id);
        w_u64(os, e->tokens.size());
        os.write((const char*)e->tokens.data(), e->tokens.size() * 4);

        w_u64(os, e->kv_main_header.size());
        if (!e->kv_main_header.empty()) os.write((const char*)e->kv_main_header.data(), e->kv_main_header.size());
        w_u32(os, (uint32_t)e->cells_main.size());
        for (const auto & c : e->cells_main) { w_u64(os, c.hash); w_u32(os, c.size); }

        w_u64(os, e->kv_drft_header.size());
        if (!e->kv_drft_header.empty()) os.write((const char*)e->kv_drft_header.data(), e->kv_drft_header.size());
        w_u32(os, (uint32_t)e->cells_drft.size());
        for (const auto & c : e->cells_drft) { w_u64(os, c.hash); w_u32(os, c.size); }

        w_u32(os, (uint32_t)e->checkpoints.size());
        for (const auto & ck : e->checkpoints) {
            w_i64(os, ck.n_tokens); w_i32(os, ck.pos_min); w_i32(os, ck.pos_max);
            w_u64(os, ck.data_tgt.size()); if (!ck.data_tgt.empty()) os.write((const char*)ck.data_tgt.data(), ck.data_tgt.size());
            w_u64(os, ck.data_dft.size()); if (!ck.data_dft.empty()) os.write((const char*)ck.data_dft.data(), ck.data_dft.size());
        }
        w_u64(os, 0); // reserved
    }
    LOG_INF("checkpoint_db: wrote manifest %zu entries\n", entries_.size());
}

// --- store / find ---

void checkpoint_db::store(
    const llama_tokens & tokens,
    const std::vector<uint8_t> & kv_main,
    const std::vector<uint8_t> & kv_drft,
    const std::list<common_prompt_checkpoint> & checkpoints,
    llama_context * ctx,
    llama_seq_id seq_id)
{
    evict();
    auto * e = create_entry(tokens, kv_main, kv_drft, checkpoints);

    if (!cfg_.disk_path.empty()) {
        write_entry_to_disk(e);

        // Content-addressed storage: hash the full blob for dedup.
        // Cell-level parsing (per-cell splitting for shared prefixes) is
        // available via llama_state_seq_parse_blob but the format has
        // edge cases (null layer tensors, v_trans layout) that need
        // proper debugging inside llama.cpp's state_write_data.
        // For now, full-blob dedup catches identical entries.
        if (!kv_main.empty()) {
            cell_ref cr;
            cr.hash = hash_bytes(kv_main.data(), kv_main.size());
            cr.size = (uint32_t)kv_main.size();
            write_cell(cr.hash, kv_main.data(), kv_main.size());
            e->cells_main.push_back(cr);
            e->kv_main_header = kv_main;
        }

        (void)ctx; (void)seq_id;

        if (cfg_.ram_capacity_mib > 0 && total_ram_ > cfg_.ram_capacity_mib * 1024ULL * 1024ULL) {
            e->kv_main.clear(); e->kv_drft.clear(); e->checkpoints.clear();
        } else {
            total_ram_ += e->disk_size;
        }
    } else {
        total_ram_ += e->disk_size;
    }
    (void)ctx; (void)seq_id;
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
        if (node->ent) { best = node->ent; res.n_matched = depth; }
    }
    if (!best) return res;

    entry * e = const_cast<entry*>(best);
    touch_lru(e);

    if (e->on_disk && e->kv_main.empty()) {
        read_entry_from_disk(e);
        if (!e->kv_main.empty()) {
            e->on_disk = false;
            total_disk_ -= e->disk_size;
            total_ram_  += e->disk_size;
        }
    }

    // Consider found if we have a match, even if KV data is empty
    res.found = true;
    res.matched_tokens = e->tokens;
    res.kv_main = e->kv_main;
    res.kv_drft = e->kv_drft;
    res.checkpoints = e->checkpoints;
    return res;
}

// --- eviction ---

void checkpoint_db::evict() {
    if (cfg_.disk_path.empty()) {
        while (total_ram_ > cfg_.ram_capacity_mib * 1024ULL * 1024ULL && !lru_.empty()) evict_one();
        return;
    }
    for (auto & e : entries_) if (!e->on_disk && !e->kv_main.empty()) write_entry_to_disk(e.get());
    while (total_disk_ > cfg_.disk_capacity_mib * 1024ULL * 1024ULL && !lru_.empty()) remove_entry_from_disk(lru_.back());
    if (!cfg_.disk_path.empty()) write_manifest();
}

void checkpoint_db::flush() {
    if (cfg_.disk_path.empty()) return;
    fs::create_directories(cfg_.disk_path + "/e");
    LOG_INF("checkpoint_db: flushing %zu entries\n", entries_.size());
    for (auto & e : entries_) write_entry_to_disk(e.get());
    write_manifest();
}

// --- entry helpers ---

checkpoint_db::entry * checkpoint_db::create_entry(const llama_tokens & tokens,
    const std::vector<uint8_t> & kv_main, const std::vector<uint8_t> & kv_drft,
    const std::list<common_prompt_checkpoint> & checkpoints)
{
    auto e = std::make_unique<entry>();
    e->id = next_id_++;
    e->tokens = tokens; e->kv_main = kv_main; e->kv_drft = kv_drft; e->checkpoints = checkpoints;
    e->disk_size = (uint32_t)(kv_main.size() + kv_drft.size());
    e->last_access = ggml_time_us();
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
        auto tok = e->tokens[i];
        if (!node->children[tok]) node->children[tok] = std::make_unique<trie_node>();
        node = node->children[tok].get();
        node->ent = e;
    }
}

void checkpoint_db::touch_lru(entry * e) {
    e->last_access = ggml_time_us();
    auto it = std::find(lru_.begin(), lru_.end(), e);
    if (it != lru_.end() && it != lru_.begin()) lru_.splice(lru_.begin(), lru_, it);
}

void checkpoint_db::evict_one() {
    if (lru_.empty()) return;
    entry * e = lru_.back();
    total_ram_ -= e->disk_size;
    if (!cfg_.disk_path.empty()) write_entry_to_disk(e);
    e->kv_main.clear(); e->kv_drft.clear(); e->checkpoints.clear();
    e->on_disk = true;
    lru_.pop_back();
    lru_.push_front(e);
}

// --- disk I/O: .kv files (legacy) + cell storage ---

void checkpoint_db::write_entry_to_disk(entry * e) {
    if (cfg_.disk_path.empty() || !e) return;
    fs::create_directories(cfg_.disk_path + "/e");

    const auto kpath = kv_path(e->id);
    std::ofstream os(kpath, std::ios::binary);
    if (!os.is_open()) { LOG_ERR("checkpoint_db: cannot write %s\n", kpath.c_str()); return; }

    w_u32(os, (uint32_t)e->kv_main.size());
    if (!e->kv_main.empty()) os.write((const char*)e->kv_main.data(), e->kv_main.size());
    w_u32(os, (uint32_t)e->kv_drft.size());
    if (!e->kv_drft.empty()) os.write((const char*)e->kv_drft.data(), e->kv_drft.size());
    w_u32(os, (uint32_t)e->checkpoints.size());
    for (const auto & ck : e->checkpoints) {
        w_i64(os, ck.n_tokens); w_i32(os, ck.pos_min); w_i32(os, ck.pos_max);
        w_u32(os, (uint32_t)ck.data_tgt.size()); if (!ck.data_tgt.empty()) os.write((const char*)ck.data_tgt.data(), ck.data_tgt.size());
        w_u32(os, (uint32_t)ck.data_dft.size()); if (!ck.data_dft.empty()) os.write((const char*)ck.data_dft.data(), ck.data_dft.size());
    }

    auto file_size = (uint32_t)os.tellp();
    if (e->on_disk) total_disk_ -= e->disk_size;
    total_disk_ += file_size;
    e->disk_size = file_size;
    e->on_disk = true;
}

void checkpoint_db::read_entry_from_disk(entry * e) const {
    if (!e || !e->on_disk) return;

    // Try .kv file first (fast path)
    auto kpath = kv_path(e->id);
    std::ifstream is(kpath, std::ios::binary);
    if (is.is_open()) {
        auto kv_main_sz = r_u32(is);
        e->kv_main.resize(kv_main_sz);
        if (kv_main_sz > 0) is.read((char*)e->kv_main.data(), kv_main_sz);
        auto kv_drft_sz = r_u32(is);
        e->kv_drft.resize(kv_drft_sz);
        if (kv_drft_sz > 0) is.read((char*)e->kv_drft.data(), kv_drft_sz);
        e->checkpoints.clear(); // don't double-load from manifest
        auto nck = r_u32(is);
        for (uint32_t i = 0; i < nck; ++i) {
            common_prompt_checkpoint ck;
            ck.n_tokens = r_i64(is); ck.pos_min = r_i32(is); ck.pos_max = r_i32(is);
            auto ts = r_u32(is); ck.data_tgt.resize(ts); if (ts > 0) is.read((char*)ck.data_tgt.data(), ts);
            auto ds = r_u32(is); ck.data_dft.resize(ds); if (ds > 0) is.read((char*)ck.data_dft.data(), ds);
            e->checkpoints.push_back(std::move(ck));
        }
        e->on_disk = false;
        return;
    }

    // Fallback: reconstruct from cells
    if (e->cells_main.empty() || e->kv_main_header.empty()) return;

    size_t total = 0;
    for (const auto & c : e->cells_main) total += c.size;
    std::vector<uint8_t> cell_buf(total);
    size_t pos = 0;
    for (const auto & c : e->cells_main) {
        if (!read_cell(c.hash, cell_buf.data() + pos, c.size)) {
            LOG_ERR("checkpoint_db: missing cell %" PRIu64 "\n", c.hash);
            e->kv_main.clear(); return;
        }
        pos += c.size;
    }

    e->kv_main = reconstruct_blob(e->kv_main_header, e->cells_main, cell_buf.data());
    if (e->kv_main.empty()) return;

    // Draft (same approach)
    if (!e->cells_drft.empty() && !e->kv_drft_header.empty()) {
        size_t dtotal = 0;
        for (const auto & c : e->cells_drft) dtotal += c.size;
        std::vector<uint8_t> dcell_buf(dtotal);
        size_t dpos = 0;
        for (const auto & c : e->cells_drft) {
            if (!read_cell(c.hash, dcell_buf.data() + dpos, c.size)) { e->kv_drft.clear(); return; }
            dpos += c.size;
        }
        e->kv_drft = reconstruct_blob(e->kv_drft_header, e->cells_drft, dcell_buf.data());
    }

    e->on_disk = false;
}

void checkpoint_db::remove_entry_from_disk(entry * e) {
    // Remove trie linkage
    if (root_) {
        trie_node * node = root_.get();
        for (size_t i = 0; i < e->tokens.size(); ++i) {
            auto it = node->children.find(e->tokens[i]);
            if (it == node->children.end()) break;
            node = it->second.get();
        }
        if (node->ent == e) node->ent = nullptr;
    }

    // Remove .kv file
    if (e->on_disk) {
        std::error_code ec; fs::remove(kv_path(e->id), ec);
        total_disk_ -= e->disk_size;
    }

    // Remove orphaned cells (reference-counted)
    std::unordered_map<uint64_t, uint32_t> refcount;
    for (const auto & other : entries_) {
        if (other.get() == e) continue;
        for (const auto & c : other->cells_main) refcount[c.hash]++;
        for (const auto & c : other->cells_drft) refcount[c.hash]++;
    }
    for (const auto & c : e->cells_main) if (refcount[c.hash] == 0) remove_cell(c.hash);
    for (const auto & c : e->cells_drft) if (refcount[c.hash] == 0) remove_cell(c.hash);

    auto it = std::find(lru_.begin(), lru_.end(), e);
    if (it != lru_.end()) lru_.erase(it);

    auto eit = std::find_if(entries_.begin(), entries_.end(),
        [e](const auto & p) { return p.get() == e; });
    if (eit != entries_.end()) { total_ram_ -= e->disk_size; entries_.erase(eit); }
}
