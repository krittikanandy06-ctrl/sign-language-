"""
Lightweight English Autocomplete and Prefix Search for Sign Language Fingerspelling.
Provides instant prefix-matching suggestions for real-time sentence construction.
"""

# Curated list of common English conversational words, pronouns, questions, and action verbs
COMMON_WORDS = [
    # Pronouns & Greetings
    "I", "ME", "MY", "YOU", "YOUR", "HE", "SHE", "IT", "WE", "THEY", "THIS", "THAT",
    "HELLO", "HI", "HEY", "GOOD", "MORNING", "EVENING", "NIGHT", "BYE", "GOODBYE",
    "PLEASE", "THANK", "THANKS", "WELCOME", "SORRY", "EXCUSE",
    
    # Common verbs
    "AM", "IS", "ARE", "WAS", "WERE", "BE", "BEEN", "HAVE", "HAS", "HAD", "DO", "DOES", "DID",
    "WANT", "NEED", "LIKE", "LOVE", "KNOW", "THINK", "SEE", "LOOK", "HEAR", "FEEL",
    "COME", "GO", "WALK", "RUN", "STOP", "START", "WAIT", "HELP", "CALL", "ASK",
    "TELL", "SAY", "SPEAK", "TALK", "WRITE", "READ", "LEARN", "TEACH", "STUDY",
    "EAT", "DRINK", "SLEEP", "WAKE", "REST", "WORK", "PLAY", "BUY", "PAY", "GIVE", "TAKE",
    
    # Question words
    "WHAT", "WHEN", "WHERE", "WHO", "WHOM", "WHOSE", "WHY", "HOW", "WHICH",
    
    # Common Nouns & Topics
    "NAME", "FRIEND", "FAMILY", "MOTHER", "FATHER", "BROTHER", "SISTER", "SON", "DAUGHTER",
    "MAN", "WOMAN", "PERSON", "PEOPLE", "CHILD", "CHILDREN", "DOCTOR", "NURSE", "POLICE",
    "TEACHER", "STUDENT", "CLASS", "SCHOOL", "HOSPITAL", "OFFICE", "HOME", "HOUSE", "ROOM",
    "WATER", "FOOD", "TEA", "COFFEE", "BREAD", "RICE", "FRUIT", "MEDICINE",
    "DAY", "TODAY", "TOMORROW", "YESTERDAY", "TIME", "HOUR", "MINUTE", "WEEK", "MONTH", "YEAR",
    "MONEY", "PHONE", "CAR", "BUS", "TRAIN", "ROAD", "CITY", "COUNTRY", "WORLD",
    
    # Adjectives & Descriptors
    "HAPPY", "SAD", "ANGRY", "TIRED", "SICK", "HEALTHY", "FINE", "OKAY", "GREAT",
    "BIG", "SMALL", "FAST", "SLOW", "HOT", "COLD", "WARM", "COOL", "NEW", "OLD",
    "EASY", "HARD", "BUSY", "FREE", "READY", "SURE", "TRUE", "RIGHT", "WRONG",
    "BEAUTIFUL", "NICE", "BAD", "CLEAN", "DIRTY", "SAFE", "DANGEROUS",
    
    # Conjunctions, Prepositions & Numbers
    "AND", "OR", "BUT", "BECAUSE", "IF", "WITH", "WITHOUT", "FOR", "FROM", "TO",
    "IN", "ON", "AT", "BY", "ABOUT", "OVER", "UNDER", "AFTER", "BEFORE",
    "YES", "NO", "NOT", "NEVER", "ALWAYS", "SOMETIMES", "MAYBE", "SOON", "NOW", "LATER",
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN"
]


class AutocompleteEngine:
    """Trie-based prefix matching engine for fast word completion."""
    
    def __init__(self, word_list=None):
        self.words = sorted(list(set(word_list or COMMON_WORDS)))
        
    def suggest(self, prefix: str, max_results: int = 4):
        """Return top word suggestions matching the given uppercase prefix."""
        if not prefix:
            return []
        
        prefix = prefix.strip().upper()
        matches = []
        
        # Exact starts-with match
        for word in self.words:
            if word.startswith(prefix) and word != prefix:
                matches.append(word)
                if len(matches) >= max_results:
                    break
                    
        return matches

    def add_word(self, word: str):
        """Add a custom word to the vocabulary."""
        word = word.strip().upper()
        if word and word not in self.words:
            self.words.append(word)
            self.words.sort()
