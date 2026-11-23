//! Joins transcripts of overlapping audio chunks by removing the words repeated at the seam.

fn norm(w: &str) -> String {
    w.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect()
}

/// Appends `new` to `acc`, dropping the longest prefix of `new` (at most 8 words) that equals the end of `acc`
/// (case and punctuation ignored). Returns how many words were dropped.
pub fn append_dedup(acc: &mut Vec<String>, new: &[String]) -> usize {
    let max = acc.len().min(new.len()).min(8);
    let mut k = 0;
    for n in (1..=max).rev() {
        if acc[acc.len() - n..].iter().zip(&new[..n]).all(|(a, b)| norm(a) == norm(b) && !norm(a).is_empty()) {
            k = n;
            break;
        }
    }
    acc.extend(new[k..].iter().cloned());
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    fn v(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }
    #[test]
    fn drops_seam_overlap() {
        let mut a = v("and so my fellow Americans,");
        assert_eq!(append_dedup(&mut a, &v("Americans ask not what")), 1);
        assert_eq!(a.join(" "), "and so my fellow Americans, ask not what");
    }
    #[test]
    fn multi_word_and_case() {
        let mut a = v("ask not what your country");
        assert_eq!(append_dedup(&mut a, &v("What YOUR country can do")), 3);
        assert_eq!(a.join(" "), "ask not what your country can do");
    }
    #[test]
    fn no_overlap_keeps_all() {
        let mut a = v("hello there");
        assert_eq!(append_dedup(&mut a, &v("general kenobi")), 0);
        assert_eq!(a.len(), 4);
    }
}
