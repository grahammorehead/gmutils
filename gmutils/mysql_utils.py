""" mysql_utils.py

Helper functions for MySQL

"""

import os, sys, re
import json
import MySQLdb

from gmutils.utils import err, argparser, isTrue

################################################################################

def mysql_connect(host="localhost", db=None, user="root", passwd=None):
    """
    Open up a cursor to a mysql db
    """
    db = MySQLdb.connect(   host=host,  # your host 
                            user=user,       # username
                            passwd=passwd,     # password
                            db=db )   # name of the database
    cur = db.cursor()
    return cur


def mysql_query(cur, query):
    """
    Run a given query against a cursor
    """
    cur.execute(query)

    for row in cur.fetchall() :
        print(row)
    return "done"


################################################################################
# MAIN

if __name__ == '__main__':

    parser = argparser({'desc': "Helper functions for MySQL: mysql_utils.py"})
    parser.add_argument('--query',  help='Query', required=False, type=str)
    parser.add_argument('--user',   help='Username', required=False, type=str)
    parser.add_argument('--passwd', help='Password', required=False, type=str)
    parser.add_argument('--db',     help='Database name', required=False, type=str)
    
    args = parser.parse_args()   # Get inputs and options

    if args.query:
        cur = mysql_connect(host=args.host, db=args.db, user=args.user, passwd=args.passwd)
        print(mysql_query(cur, args.query))

    
################################################################################
################################################################################
