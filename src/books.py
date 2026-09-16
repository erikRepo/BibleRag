"""Standard 66-book numbering (1-39 Old Testament, 40-66 New Testament)
used by the Holy-Bible-XML-Format repo's XML files."""

BOOK_NAMES = {
    1: "Genesis", 2: "Exodus", 3: "Leviticus", 4: "Numbers", 5: "Deuteronomy",
    6: "Joshua", 7: "Judges", 8: "Ruth", 9: "1 Samuel", 10: "2 Samuel",
    11: "1 Kings", 12: "2 Kings", 13: "1 Chronicles", 14: "2 Chronicles",
    15: "Ezra", 16: "Nehemiah", 17: "Esther", 18: "Job", 19: "Psalms",
    20: "Proverbs", 21: "Ecclesiastes", 22: "Song of Solomon", 23: "Isaiah",
    24: "Jeremiah", 25: "Lamentations", 26: "Ezekiel", 27: "Daniel",
    28: "Hosea", 29: "Joel", 30: "Amos", 31: "Obadiah", 32: "Jonah",
    33: "Micah", 34: "Nahum", 35: "Habakkuk", 36: "Zephaniah", 37: "Haggai",
    38: "Zechariah", 39: "Malachi",
    40: "Matthew", 41: "Mark", 42: "Luke", 43: "John", 44: "Acts",
    45: "Romans", 46: "1 Corinthians", 47: "2 Corinthians", 48: "Galatians",
    49: "Ephesians", 50: "Philippians", 51: "Colossians",
    52: "1 Thessalonians", 53: "2 Thessalonians", 54: "1 Timothy",
    55: "2 Timothy", 56: "Titus", 57: "Philemon", 58: "Hebrews",
    59: "James", 60: "1 Peter", 61: "2 Peter", 62: "1 John", 63: "2 John",
    64: "3 John", 65: "Jude", 66: "Revelation",
}

BOOK_NAMES_FI = {
    1: "1. Mooseksen kirja", 2: "2. Mooseksen kirja", 3: "3. Mooseksen kirja",
    4: "4. Mooseksen kirja", 5: "5. Mooseksen kirja", 6: "Joosua", 7: "Tuomarien kirja",
    8: "Ruut", 9: "1. Samuelin kirja", 10: "2. Samuelin kirja", 11: "1. Kuningasten kirja",
    12: "2. Kuningasten kirja", 13: "1. Aikakirja", 14: "2. Aikakirja", 15: "Esra",
    16: "Nehemia", 17: "Ester", 18: "Job", 19: "Psalmit", 20: "Sananlaskut",
    21: "Saarnaaja", 22: "Laulujen laulu", 23: "Jesaja", 24: "Jeremia",
    25: "Valitusvirret", 26: "Hesekiel", 27: "Daniel", 28: "Hoosea", 29: "Jooel",
    30: "Aamos", 31: "Obadja", 32: "Joona", 33: "Miika", 34: "Nahum", 35: "Habakuk",
    36: "Sefanja", 37: "Haggai", 38: "Sakarja", 39: "Malakia",
    40: "Matteus", 41: "Markus", 42: "Luukas", 43: "Johannes", 44: "Apostolien teot",
    45: "Roomalaiskirje", 46: "1. Korinttilaiskirje", 47: "2. Korinttilaiskirje",
    48: "Galatalaiskirje", 49: "Efesolaiskirje", 50: "Filippiläiskirje",
    51: "Kolossalaiskirje", 52: "1. Tessalonikalaiskirje", 53: "2. Tessalonikalaiskirje",
    54: "1. Timoteuskirje", 55: "2. Timoteuskirje", 56: "Titukselle", 57: "Filemonille",
    58: "Heprealaiskirje", 59: "Jaakobin kirje", 60: "1. Pietarin kirje",
    61: "2. Pietarin kirje", 62: "1. Johanneksen kirje", 63: "2. Johanneksen kirje",
    64: "3. Johanneksen kirje", 65: "Juudaksen kirje", 66: "Ilmestyskirja",
}

BOOK_NAMES_BY_LANG = {"en": BOOK_NAMES, "fi": BOOK_NAMES_FI}

BOOK_NUMBERS_BY_NAME = {name: num for num, name in BOOK_NAMES.items()}
