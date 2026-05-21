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
    std::string disk_path = "";            // empty = disk tier disabled
    uint64_t disk_capacity_mib = 102400;   // 100 GiB default
    uint64_t ram_capacity_mib = 4096;      // 4 GiB for RAM-resident entries
};

class checkpoint_db {
public:
    struct match_result {
        bool found = false;
        size_t n_matched = 0;                           // LCP length
        llama_tokens matched_tokens;
        std::vector<uint8_t> kv_main;
        std::vector<uint8_t> kv_drft;
        std::list<common_prompt_checkpoint> checkpoints;
    };

    checkpoint_db(const checkpoint_db_config & cfg);
    ~checkpoint_db();

    // rebuild trie from disk manifest (call on server startup after model load)
    void load_manifest();

    // store a new prefix-KV mapping (call after prompt_save)
    // tokens, kv_main, kv_drft are copied; checkpoints are copied
    void store(
        const llama_tokens & tokens,
        const std::vector<uint8_t> & kv_main,
        const std::vector<uint8_t> & kv_drft,
        const std::list<common_prompt_checkpoint> & checkpoints);

    // find best LCP match for a query prompt
    match_result find(const llama_tokens & query);

    // evict entries to stay under capacity limits (LRU)
    void evict();

    // flush trie → disk (called on server shutdown)
    void flush();

    // stats
    size_t n_entries() const { return entries_.size(); }
    uint64_t disk_size() const { return total_disk_; }
    uint64_t ram_size() const  { return total_ram_; }

private:
    struct entry {
        uint32_t id;
        llama_tokens tokens;
        std::vector<uint8_t> kv_main;
        std::vector<uint8_t> kv_drft;
        std::list<common_prompt_checkpoint> checkpoints;

        // disk backing
        bool on_disk = false;
        uint64_t disk_offset = 0;
        uint32_t disk_size   = 0;

        // lru
        mutable int64_t last_access;
    };

    struct trie_node {
        std::unordered_map<llama_token, std::unique_ptr<trie_node>> children;
        entry * ent = nullptr;
    };

    checkpoint_db_config cfg_;
    std::unique_ptr<trie_node> root_;
    std::vector<std::unique_ptr<entry>> entries_;
    std::list<entry*> lru_;

    uint64_t total_ram_  = 0;
    uint64_t total_disk_ = 0;
    uint32_t next_id_    = 1;

    // helpers
    void insert_into_trie(entry * e);
    entry * create_entry(
        const llama_tokens & tokens,
        const std::vector<uint8_t> & kv_main,
        const std::vector<uint8_t> & kv_drft,
        const std::list<common_prompt_checkpoint> & checkpoints);
    void touch_lru(entry * e);
    void evict_one();

    // disk I/O
    std::string manifest_path() const;
    std::string kv_path(uint32_t id) const;
    void write_entry_to_disk(entry * e);
    void read_entry_from_disk(entry * e) const;
    void remove_entry_from_disk(entry * e);
    void read_manifest();
    void write_manifest();
};
