
import sqlite3
import sys
sys.path = [r'C:\Users\owen\Documents\Projects\torch4nlp'] + sys.path
print(sys.path)

import json
import re
from flask import Flask, render_template  
from torch4nlp import *


app = Flask(__name__)


def query_blog(idx=None):
    try:
        conn = sqlite3.connect('youngnlp.db')
        c = conn.cursor()
        if idx is None:
            query = 'select * from blog'
        else:
            query = 'select * from blog where blog_id="{}"'.format(idx)
        # print(query)
        res = c.execute(query)
        res = res.fetchall()
        conn.close()
        return res
    except Exception as e:
        print('query blog raise exception: {}'.format(e))


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/cs')
def cs():
    blogs = query_blog()
    blog_info = []
    for x in blogs:
        blog_info.append({'id': x[0], 'title': x[1]})

    return render_template('cs/cs.html', blog_info=blog_info)


@app.route('/economy')
def economy():
    return render_template('economy.html')


@app.route('/cs/kg')
def read_data():
    res = []
    for x in read_json(r'C:\Users\owen\Documents\Projects\youngfoo.github.io\baidubaike_240719.json'):
        title = x['title']
        title = re.sub(r'（[^（）]+）', '', x['title'])
        title = title.replace('_百度百科', '')
        res.append(title)
        # print(title)
    res = sorted(res)
    return render_template('cs/data.html', data_list=res)


@app.route('/cs/blog/<blog_id>')
def display_cs_blog(blog_id):
    try:
        print('start to connect to database')
        conn = sqlite3.connect('youngnlp.db')
        c = conn.cursor()
        query = 'select * from blog where blog_id="{}"'.format(blog_id)
        print(query)
        c.execute(query)
        res = c.fetchone()
        print('res: {}'.format(res))
        c.close()
        conn.close()
        return render_template('cs/blog.html', blog_id=res[0], blog_title=res[1], blog_content=res[3])
    except Exception as e:
        return render_template('404.html', msg='failed to fetch blog content(blog_id={})'.format(blog_id))


# 运行Flask应用  
if __name__ == '__main__':
    app.run(debug=True)
