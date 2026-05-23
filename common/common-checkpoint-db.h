#pragma once

#include "common.h"
#include "llama.h"

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct common_prompt_checkpoint;

struct checkpoint_db_config {
    std::string disk_path = "";
    uint64_t disk_capacity_mib = 102400;
    uint64_t ram_capacity_mib = 4096;
};

class checkpoint_db {
public:
    struct match_result {
        bool found = false;
        size_t n_matched = 0;
        llama_tokens matched_tokens;
        std::vector<uint8_t> kv_main;
        std::vector<uint8_t> kv_drft;
        std::list<common_prompt_checkpoint> checkpoints;
    };

    checkpoint_db(const checkpoint_db_config & cfg);
    ~checkpoint_db();

    void set_model_id(const std::string & id) { model_id_ = id; }

    void load_manifest();

    // store — now takes optional context + seq_id for cell-based dedup storage
    void store(
        const llama_tokens & tokens,
        const std::vector<uint8_t> & kv_main,
        const std::vector<uint8_t> & kv_drft,
        const std::list<common_prompt_checkpoint> & checkpoints,
        llama_context * ctx = nullptr,
        llama_seq_id seq_id = 0);

    match_result find(const llama_tokens & query);

    void evict();
    void flush();

    size_t n_entries() const { return entries_.size(); }
    uint64_t disk_size() const { return total_disk_; }
    uint64_t ram_size() const  { return total_ram_; }

private:
    struct cell_ref {
        uint64_t hash;
        uint32_t size;
    };

    struct entry {
        uint32_t id;
        llama_tokens tokens;
        std::vector<uint8_t> kv_main;
        std::vector<uint8_t> kv_drft;
        std::list<common_prompt_checkpoint> checkpoints;

        // cell-based dedup storage (content-addressed, shared across entries)
        std::vector<uint8_t> kv_main_header;
        std::vector<cell_ref> cells_main;
        std::vector<uint8_t> kv_drft_header;
        std::vector<cell_ref> cells_drft;

        bool on_disk = false;
        uint32_t disk_size = 0;
        mutable int64_t last_access;
    };

    struct trie_node {
        std::unordered_map<llama_token, std::unique_ptr<trie_node>> children;
        entry * ent = nullptr;
    };

    checkpoint_db_config cfg_;
    std::string model_id_;
    std::unique_ptr<trie_node> root_;
    std::vector<std::unique_ptr<entry>> entries_;
    std::list<entry*> lru_;

    uint64_t total_ram_  = 0;
    uint64_t total_disk_ = 0;
    uint32_t next_id_    = 1;

    static uint64_t hash_bytes(const uint8_t * data, size_t len);
    static std::vector<uint8_t> reconstruct_blob(
        const std::vector<uint8_t> & header,
        const std::vector<cell_ref> & cells,
        const uint8_t * cell_data_buf);
    std::string cell_path(uint64_t hash) const;
    void write_cell(uint64_t hash, const uint8_t * data, size_t len);
    bool read_cell(uint64_t hash, uint8_t * out, size_t len) const;
    void remove_cell(uint64_t hash);

    void insert_into_trie(entry * e);
    entry * create_entry(
        const llama_tokens & tokens,
        const std::vector<uint8_t> & kv_main,
        const std::vector<uint8_t> & kv_drft,
        const std::list<common_prompt_checkpoint> & checkpoints);
    void touch_lru(entry * e);
    void evict_one();

    std::string manifest_path() const;
    std::string kv_path(uint32_t id) const;
    void write_entry_to_disk(entry * e);
    void read_entry_from_disk(entry * e) const;
    void remove_entry_from_disk(entry * e);
    void write_manifest();
    void load_manifest_v2(std::ifstream & is, uint32_t n);
    void load_manifest_v3(std::ifstream & is, uint32_t n);
};
