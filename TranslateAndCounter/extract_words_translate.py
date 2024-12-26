import nltk
from ebooklib import epub
from bs4 import BeautifulSoup
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from collections import defaultdict,Counter

# 下载必要的 nltk 数据集
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
# 初始化词形还原器
lemmatizer = WordNetLemmatizer()

# 提取 epub 内容为纯文本
def extract_text_from_epub(file_path):
    book = epub.read_epub(file_path)
    full_text = ""
    for item in book.items:  # 直接迭代 book.items
        if item.media_type == "application/xhtml+xml":  # 检查 MIME 类型
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            full_text += soup.get_text() + "\n"

    return full_text
# 根据词频和常用性划分单词级别
# 使用 nltk 自带的词库或词频信息
from nltk.corpus import words
nltk.download('words')

# 根据词频和常用性划分单词级别
# 使用 nltk 自带的词库或词频信息
def categorize_words(text):
    text_tokens = nltk.word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in text_tokens if word.isalpha()]
    word_counts = Counter(lemmatized_tokens)  # 统计词频
    word_levels = defaultdict(list)

    for word, count in word_counts.items():
        if word in nltk.corpus.words.words():
            word_levels["Common"].append((word, count))
        else:
            word_levels["Advanced"].append((word, count))
    return word_levels

def get_word_definition(word_tuple):
    word = word_tuple[0]  # 从元组中提取单词
    synsets = wordnet.synsets(word)
    if synsets:
        return synsets[0].definition()  # 返回第一个定义
    return "No definition found."

# 主程序：提取文本、分类单词并生成解释
def main(epub_path):
    # 提取 epub 文本
    text = extract_text_from_epub(epub_path)
    # 按级别分类单词
    word_levels = categorize_words(text)

    # 输出单词及其定义
    for level, words in word_levels.items():
        print(f"\n{level} words and definitions:")
        for word_tuple in sorted(words, key=lambda x: x[1], reverse=True):  # 按词频从高到低排序
            definition = get_word_definition(word_tuple)
            print(f"{word_tuple[0]} ({word_tuple[1]} occurrences): {definition}")
# 示例调用
if __name__ == "__main__":
    # 请将以下路径替换为实际文件路径
    epub_path = "C:/Users/XX/Downloads/The Notebook (Nicholas Sparks) (Z-Library).epub"

    main(epub_path)
