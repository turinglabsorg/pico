use anyhow::Result;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::mt::{Lang, OllamaTranslator};

const CACHE_MAX_ENTRIES: usize = 500;
const CACHE_TTL_MS: u128 = 10 * 60 * 1000; // 10 min

#[derive(Clone)]
struct CacheKey {
    src: Lang,
    tgt: Lang,
    text: String,
}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.src == other.src && self.tgt == other.tgt && self.text == other.text
    }
}

impl Eq for CacheKey {}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.src, self.tgt, &self.text).hash(state);
    }
}

#[derive(Clone)]
struct CacheEntry {
    result: String,
    last_used: Instant,
    hits: u64,
}

pub struct MtCache {
    translator: OllamaTranslator,
    store: Arc<Mutex<HashMap<CacheKey, CacheEntry>>>,
    hits: Arc<Mutex<u64>>,
    misses: Arc<Mutex<u64>>,
}

impl MtCache {
    pub fn new(translator: OllamaTranslator) -> Self {
        Self {
            translator,
            store: Arc::new(Mutex::new(HashMap::with_capacity(CACHE_MAX_ENTRIES))),
            hits: Arc::new(Mutex::new(0)),
            misses: Arc::new(Mutex::new(0)),
        }
    }

    fn get(&self, text: &str, src: Lang, tgt: Lang) -> Option<String> {
        let key = CacheKey {
            src,
            tgt,
            text: Self::normalize(text),
        };
        let mut store = self.store.lock().ok()?;
        let entry = store.get_mut(&key)?;
        if entry.last_used.elapsed().as_millis() > CACHE_TTL_MS {
            store.remove(&key);
            return None;
        }
        entry.last_used = Instant::now();
        entry.hits += 1;
        let result = entry.result.clone();
        drop(store);
        if let Ok(mut hits) = self.hits.lock() {
            *hits += 1;
        }
        Some(result)
    }

    fn put(&self, text: &str, src: Lang, tgt: Lang, result: String) {
        let key = CacheKey {
            src,
            tgt,
            text: Self::normalize(text),
        };
        if let Ok(mut store) = self.store.lock() {
            store.insert(
                key,
                CacheEntry {
                    result,
                    last_used: Instant::now(),
                    hits: 1,
                },
            );
            if store.len() > CACHE_MAX_ENTRIES {
                let oldest = store
                    .iter()
                    .min_by_key(|(_, v)| v.last_used)
                    .map(|(k, _)| k.clone());
                if let Some(k) = oldest {
                    store.remove(&k);
                }
            }
        }
    }

    /// Translate with caching layer. Falls back to the underlying translator on miss.
    pub async fn translate(&self, text: &str, src: Lang, tgt: Lang) -> Result<String> {
        if let Some(cached) = self.get(text, src, tgt) {
            return Ok(cached);
        }
        let translated = self.translator.translate(text, src, tgt).await?;
        self.put(text, src, tgt, translated.clone());
        if let Ok(mut m) = self.misses.lock() {
            *m += 1;
        }
        Ok(translated)
    }

    pub fn stats(&self) -> (u64, u64, usize) {
        let h = *self.hits.lock().unwrap();
        let m = *self.misses.lock().unwrap();
        let len = self.store.lock().unwrap().len();
        (h, m, len)
    }

    /// Normalize for cache key: lowercase, collapse whitespace, trim punctuation.
    fn normalize(s: &str) -> String {
        s.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_cache() -> MtCache {
        let tr = OllamaTranslator::new(
            "http://localhost:0".to_string(),
            "test-model".to_string(),
            None,
        )
        .expect("translator builds");
        MtCache::new(tr)
    }

    #[test]
    fn cache_hit_same_text() {
        let cache = dummy_cache();
        cache.put("Hello world", Lang::En, Lang::Fr, "Bonjour le monde".to_string());
        let hit = cache.get("hello world", Lang::En, Lang::Fr);
        assert_eq!(hit, Some("Bonjour le monde".to_string()));
    }

    #[test]
    fn cache_miss_wrong_lang() {
        let cache = dummy_cache();
        cache.put("Hello", Lang::En, Lang::Fr, "Bonjour".to_string());
        let miss = cache.get("hello", Lang::En, Lang::Es);
        assert!(miss.is_none());
    }

    #[test]
    fn cache_normalization() {
        let cache = dummy_cache();
        cache.put("  Hello,   WORLD!!  ", Lang::En, Lang::Fr, "Salut".to_string());
        let hit = cache.get("hello world", Lang::En, Lang::Fr);
        assert_eq!(hit, Some("Salut".to_string()));
    }

    #[test]
    fn cache_eviction_happens() {
        let cache = dummy_cache();
        for i in 0..(CACHE_MAX_ENTRIES + 10) {
            let text = format!("unique entry {}", i);
            cache.put(&text, Lang::En, Lang::Fr, format!("fr {}", i));
        }
        let len = cache.store.lock().unwrap().len();
        assert!(len <= CACHE_MAX_ENTRIES, "expected eviction, got {}", len);
    }

    #[test]
    fn cache_hits_counter() {
        let cache = dummy_cache();
        cache.put("test", Lang::En, Lang::Fr, "fr".to_string());
        cache.get("test", Lang::En, Lang::Fr);
        cache.get("test", Lang::En, Lang::Fr);
        let (h, _m, _) = cache.stats();
        assert_eq!(h, 2);
    }
}
