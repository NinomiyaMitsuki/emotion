#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 @desc:
 @author: Yu Chao
 @contact: yucnb@cn.ibm.com
 @software: PyCharm  @since:python 3.5.2
 """

import logging

logging.basicConfig(level=logging.INFO,
                    filename='log.txt',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

def main():
    logging.info('main module start')
    logging.info('main module stop')
    try:
        raise RuntimeError("Bad hostname")
    except RuntimeError as e:
        logging.error(e)


if __name__ == '__main__':
    main()
