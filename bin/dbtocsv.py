#!/usr/bin/env python

import csv
import argparse
import MySQLdb as mysql
from itertools import count

window_size = 1000


def main(args):
    db = mysql.connect(host=args.host, user=args.user, passwd=args.password)

    def fieldnames():
        query = f'DESCRIBE reclameaqui.complaint;'
        cur = db.cursor()
        cur.execute(query)
        results = cur.fetchall()
        return [c[0] for c in results]

    def windows_query():
        for c in count():
            lb = c * window_size
            up = lb + window_size - 1
            query = f'SELECT * FROM reclameaqui.complaint ' \
                    f'where id >= {lb} and id < {up};'
            cur = db.cursor()
            cur.execute(query)
            results = cur.fetchall()
            if not results:
                break
            for r in results:
                yield {x: str(y) for x, y in zip(columns, r)}

    def normalize_values():
        def f(v):
            if v == 10:
                return 'satisfied'
            elif v == 5:
                return 'neutral'
            elif v == 0:
                return 'not satisfied'
            else:
                return False

        for result in windows_query():
            result['rate'] = f(int(result['rate']))
            if result['rate'] == False:
                continue
            yield result

    columns = fieldnames()
    with open('input.csv', 'w') as h:
        writer = csv.DictWriter(h, columns)
        writer.writeheader()
        for result in normalize_values():
            writer.writerow(result)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--user', required=True)
    parser.add_argument('--password', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())