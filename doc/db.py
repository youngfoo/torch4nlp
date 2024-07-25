import json
import sqlite3
import uuid


def insert_blog(idx, title, tag, content):
    try:
        conn = sqlite3.connect('youngnlp.db')
        c = conn.cursor()
        cmd = 'INSERT INTO blog(blog_id, blog_title, blog_tag, blog_content) VALUES ("{}", "{}", "{}", {})'.format(
            idx, title, tag, json.dumps(content, ensure_ascii=False))
        print(cmd)
        c.execute(cmd)
        print(c.rowcount)
        conn.commit()
        c.close()
        conn.close()
    except Exception as e:
        print('create blog raise exception: {}'.format(e))


def delete_blog(blog_name):
    try:
        conn = sqlite3.connect('youngnlp.db')
        cur = conn.cursor()
        query = "DROP TABLE IF EXISTS {}".format(blog_name)
        cur.execute(query)
        conn.commit()
        cur.close()
        conn.close()
        print('delete blog successfully!')
    except Exception as e:
        print('delete blog raise exception: {}'.format(e))
        


insert_blog(
    uuid.uuid1().hex,
    'test2',
    '',
    open('a.md', 'r', encoding='utf8').read()
)