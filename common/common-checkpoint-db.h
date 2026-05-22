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
        size_t n_matched = 0;
        llama_tokens matched_tokens;
        std::vector<uint8_t> kv_main;
        std::vector<uint8_t> kv_drft;
        std::list<common_prompt_checkpoint> checkpoints;
    };

    checkpoint_db(const checkpoint_db_config & cfg);
    ~checkpoint_db();

    void set_model_id(const std::string & id) { model_id_ = id; }

    // tell the DB how many bytes each token position occupies in the serialized
    // KV blob — enables position-aligned chunk boundaries for better dedup.
    // call after set_model_id, before any store/find.
    // pass 0 to fall back to fixed-size chunks.
    void set_bytes_per_token(size_t bpt) { bytes_per_token_ = bpt; }

    void load_manifest();

    void store(
        const llama_tokens & tokens,
        const std::vector<uint8_t> & kv_main,
        const std::vector<uint8_t> & kv_drft,
        const std::list<common_prompt_checkpoint> & checkpoints);

    match_result find(const llama_tokens & query);

    void evict();
    void flush();

    size_t n_entries() const { return entries_.size(); }
    uint64_t disk_size() const { return total_disk_; }
    uint64_t ram_size() const  { return total_ram_; }

private:
    // a content-addressed chunk reference
    struct chunk_ref {
        uint64_t hash;    // XXH3_64 of chunk data
        uint32_t size;    // bytes
    };

    struct entry {
        uint32_t id;
        llama_tokens tokens;
        std::vector<uint8_t> kv_main;
        std::vector<uint8_t> kv_drft;
        std::list<common_prompt_checkpoint> checkpoints;

        // content-addressed chunk references (main + drft)
        std::vector<chunk_ref> chunks_main;
        std::vector<chunk_ref> chunks_drft;

        bool on_disk = false;
        uint32_t disk_size = 0;           // total chunk data bytes
        mutable int64_t last_access;
    };

    struct trie_node {
        std::unordered_map<llama_token, std::unique_ptr<trie_node>> children;
        entry * ent = nullptr;
    };

    static const size_t CHUNK_TARGET = 65536; // 64 KB nominal chunk size

    checkpoint_db_config cfg_;
    std::string model_id_;
    size_t bytes_per_token_ = 0;             // 0 = fixed chunks
    std::unique_ptr<trie_node> root_;
    std::vector<std::unique_ptr<entry>> entries_;
    std::list<entry*> lru_;

    uint64_t total_ram_  = 0;
    uint64_t total_disk_ = 0;
    uint32_t next_id_    = 1;

    // chunk helpers
    static uint64_t hash_chunk(const uint8_t * data, size_t len);
    std::string chunk_path(uint64_t hash) const;
    void write_chunk(uint64_t hash, const uint8_t * data, size_t len);
    bool read_chunk(uint64_t hash, uint8_t * out, size_t len) const;
    void remove_chunk(uint64_t hash);
    std::vector<chunk_ref> chunkify(const uint8_t * data, size_t data_size) const;
    std::vector<uint8_t> dechunkify(const std::vector<chunk_ref> & refs) const;

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
    std::string chunks_dir() const;
    void write_entry_chunks(entry * e);
    void read_entry_chunks(entry * e) const;
    void remove_entry_chunks(entry * e);
    void write_manifest();
    void load_manifest_v3(std::ifstream & is, uint32_t n);
};
