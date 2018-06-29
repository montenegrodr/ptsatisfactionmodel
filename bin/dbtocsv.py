#!/usr/bin/env python

import csv
import argparse
import MySQLdb as mysql
from itertools import count

window_size = 1000


def main(args):
    db = mysql.connect(host=args.host, user=args.user, passwd=args.password)

    def windows_query():
        for c in count():
            lb = c * window_size
            up = lb + window_size - 1
            query = f'SELECT * FROM reclameaqui.complaint ' \
                    f'where id >= {lb} and id < {up};'
            cur = db.cursor()
            cur.execute(query)
            results = cur.fetchall()
            header = [d[0] for d in cur.description]
            if not results:
                break
            yield header
            for r in results:
                yield r

    with open('input.csv', 'w') as h:
        writer = csv.writer(h)
        for result in windows_query():
            writer.writerow(list(map(str, result)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--user', required=True)
    parser.add_argument('--password', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())