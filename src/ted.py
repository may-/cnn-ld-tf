# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs

from collections import defaultdict
import urllib
import os


def load_titles(data_dir):
    """
    Load ted talk titles from the web/files.
    Based on https://github.com/ajinkyakulkarni14/How-I-Extracted-TED-talks-for-parallel-Corpus-
    """

    def get_talk_titles(path, dic):
        r = urllib.urlopen(path).read()
        talks = bs(r, "html.parser").find_all("a", class_='')
        for talk in talks:
            href = talk.attrs['href']
            if href.find('/talks/') == 0 and not dic.get(href[7:]):
                dic[href[7:]] = True
        return dic

    print(("\tLoading ted talk titles..."))
    title_file = os.path.join(data_dir, 'TED_TALK_TITLES.csv')
    if os.path.exists(title_file):
        titles_df = pd.read_csv(title_file, sep='\t', header=None, names=['title'], index_col=0)

    else:
        all_talk_titles = defaultdict(bool)
        for i in range(1, 64):
            path = "https://www.ted.com/talks?page=%d" % (i)
            all_talk_titles = get_talk_titles(path, all_talk_titles)

        all_talk_titles = [k for k, v in all_talk_titles.iteritems() if v == True]

        title_df = pd.DataFrame(all_talk_titles, columns=['title'])
        title_df.to_csv(title_file, sep='\t', header=False)

    print(('\t\t%d titles found.' % len(titles_df.titles)))
    return titles_df


def load_transcriptions(data_dir):
    """
    Load ted talk transcriptions from the web/files.
    Based on https://github.com/ajinkyakulkarni14/How-I-Extracted-TED-talks-for-parallel-Corpus-
    """

    def extract_transcriptions(talk_title, data_path):
        path1 = "https://www.ted.com/talks/%s/transcript" % talk_title
        r1 = urllib.urlopen(path1).read()
        soup1 = bs(r1, "html.parser")
        df1 = pd.DataFrame()
        for i in soup1.findAll('link'):
            if i.get('href') != None and i.attrs['href'].find('?language=') != -1:
                lang = i.attrs['hreflang']
                path2 = i.attrs['href']
                r2 = urllib.urlopen(path2).read()
                soup2 = bs(r2, "html.parser")
                time_frame = []
                text_talk = []

                for j in soup2.findAll('span', class_='talk-transcript__fragment'):
                    time_frame.append(j.attrs['data-time'])
                    text_talk.append(j.text.replace('\n', ' '))

                df2 = pd.DataFrame()
                df2[lang] = text_talk
                df2[lang + '_time_frame'] = time_frame
                df1 = pd.concat([df1, df2], axis=1)
        df1.to_csv(os.path.join(data_path, 'orig', '%s.csv' % talk_title), sep='\t', encoding='utf-8')

    if not os.path.exists(os.path.join(data_dir, 'orig')):
        os.mkdir(os.path.join(data_dir, 'orig'))

    # Load titles
    title_df = load_titles(data_dir)
    files = []
    print(("\tLoading ted talk transcriptions..."))
    for doc_id, row in title_df.iterrows():
        title = row['title']
        orig_path = os.path.join(data_dir, 'orig', '%s.csv' % title)
        if not os.path.exists(orig_path):
            extract_transcriptions(title, data_dir)
        orig_df = pd.read_csv(orig_path, sep='\t', encoding='utf-8', index_col=0)
        orig_df['sent_id'] = orig_df.index
        orig_df['doc_id'] = pd.Series([doc_id] * len(orig_df), index=orig_df.index)

        files.append(orig_df)
    df = pd.concat(files, ignore_index=True)

    # languages list
    #languages = [c for c in df.columns.values if not c.endswith('_time_frame') and not in ['sent_id', 'doc_id']]
    languages = []

    # Save files by language
    for lang in df:
        if lang.endswith('_time_frame') or lang in ['doc_id', 'sent_id']:
            continue
        languages.append(lang)
        path = os.path.join(data_dir, 'raw', '%s.csv' % lang)
        if not os.path.exists(path):
            filtered = df[pd.notnull(df[lang])][lang]
            filtered.to_csv(path, sep='\t', encoding='utf-8', index=False, header=False)

    print(('\t\t%d languages extracted.' % len(languages)))


def main():
    limit = 500
    data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data', 'ted%d' % limit)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    load_transcriptions(data_dir)

    #language_codes = pd.read_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'language_codes.csv'), sep='\t',
    #                          skip_blank_lines=True, comment='#', header=None, names=['code', 'language'])
    #languages = language_codes.keys()

    languages = ["ar", "az", "bg", "bn", "bo", "cs", "da", "de", "el", "en", "es",
                 "fa", "fi", "fil", "fr", "gu", "he", "hi", "ht", "hu", "hy", "id",
                 "is", "it", "ja", "ka", "km", "kn", "ko", "ku", "lt", "mg", "ml",
                 "mn", "ms", "my", "nb", "ne", "nl", "nn", "pl", "ps", "pt", "ro",
                 "ru", "si", "sk", "sl", "so", "sq", "sv", "sw", "ta", "te", "tg",
                 "th", "tl", "tr", "ug", "uk", "ur", "uz", "vi", "zh-cn", "zh-tw"]

    for lang in languages:
        in_path = os.path.join(data_dir, 'raw', '%s.csv' % lang)
        out_path = os.path.join(data_dir, 'ted_%d.%s' % (limit, lang))
        df = pd.read_csv(in_path, sep='\t', encoding='utf-8', header=None, names=['doc_id', 'sent_id', 'x'])
        df = df.iloc[np.random.permutation(len(df))]
        df.x.head(limit).to_csv(out_path, sep='\t', encoding='utf-8', header=None, index=None)


if __name__ == '__main__':
    main()
