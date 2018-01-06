from nltk.stem.snowball import SnowballStemmer
import string


def norm_text(title):
    """
    given the contents of the title remove punctuation and performs stemming to return a
    string that contains important constituents of the title.
    """

    words = ""
    title = title.lower()
    title = title.translate(str.maketrans("", "", string.punctuation))

    arr = title.split()
    stemmer = SnowballStemmer("english")

    for i in range(len(arr)):
        words = words + " " + stemmer.stem(arr[i])

    return words


if __name__ == '__main__':
    main()
