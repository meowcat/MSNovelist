# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 17:12:09 2021

@author: stravsm
"""


import sqlite3
from sqlite3 import Error
from .fp_database import *
import re


def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None

class FpDatabaseSqlite(FpDatabase):
    def __init__(self, db_file, config):
        super().__init__(db_file, None) 
        
        config_ = {
            'use_regex': True
            }
        config_.update(config)
        self.config = config_
        
        self.use_regex = self.config['use_regex']
        
        self._create_connection(db_file)
        self._create_schema()
    
    SCHEMA_DEF = {
            'compounds' :
                (
            ('id', 'INTEGER PRIMARY KEY'),
            ('fingerprint', 'BLOB'),
            ('fingerprint_degraded', 'BLOB'),
            ('smiles_generic', 'CHAR(128)'),
            ('smiles_canonical', 'CHAR(128)'),
            ('inchikey', 'CHAR(27)'),
            ('inchikey1', 'CHAR(14)'),
            ('mol', 'BLOB'),
            ('mf', 'BLOB'),
            ('mf_text', 'CHAR(128)'),
            ('source', 'CHAR(128)'),
            ('grp', 'CHAR(128)'),
            ('perm_order', 'INT')
                 )
                }
    
    # Copied from tutorial!
    def _create_connection(self, db_file):
        """ create a database connection to a SQLite database """
        try:
            self._db_con = sqlite3.connect(db_file)
            self._db_con.create_function("REGEXP", 2, regexp)
            self._db_con.row_factory = sqlite3.Row
        except Error as e:
            print(e)
            
    def _create_table(cls, conn, table_name, table_def):
        # build SQL query from table definition
        sql_str = 'CREATE TABLE IF NOT EXISTS '
        sql_str = sql_str + table_name + ' ('
        table_spec = [' '.join(table_tpl) for table_tpl in table_def]
        sql_str = sql_str + ', '.join(table_spec) + ' )'
        if conn is None:
            return(sql_str)
        else:
            try:
                c = conn.cursor()
                c.execute(sql_str)
            except Error as e:
                print(e)
    
    def _create_schema(self):
        for (table_name, table_schema) in FpDatabaseSqlite.SCHEMA_DEF.items():
            self._create_table(self._db_con, table_name, table_schema)
            
    def insert_fp(self, fp_item):
        sql_str = 'INSERT INTO compounds (smiles_generic, smiles_canonical, inchikey1, inchikey, fingerprint, fingerprint_degraded, mol, mf, mf_text, source, grp) VALUES (?,?,?,?,?,?,?,?,?,?,?)'
        sql_args = [fp_item.smiles_generic,
                    fp_item.smiles_canonical,
                    fp_item.inchikey1, fp_item.inchikey, 
                    fp_item.fp, 
                    fp_item.fp_degraded, 
                    sqlite3.Binary(pickle.dumps(fp_item.mol)),
                    sqlite3.Binary(pickle.dumps(fp_item.mf)),
                    fp_item.mf_text,
                     fp_item.source, fp_item.grp]
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                cursor.execute(sql_str, sql_args)
            return cursor.lastrowid
        except Error as e:
            print(e)
    
    def insert_fp_multiple(self, fp_items):
        sql_str = 'INSERT INTO compounds (smiles_generic, smiles_canonical, inchikey1, inchikey, fingerprint, fingerprint_degraded, mol, mf, mf_text, source, grp) VALUES (?,?,?,?,?,?,?,?,?,?,?)'
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                for fp_item in fp_items:
                    sql_args = [fp_item.smiles_generic,
                                fp_item.smiles_canonical,
                                fp_item.inchikey1, fp_item.inchikey, 
                                fp_item.fp, 
                                fp_item.fp_degraded, 
                                sqlite3.Binary(pickle.dumps(fp_item.mol)),
                                sqlite3.Binary(pickle.dumps(fp_item.mf)),
                                fp_item.mf_text,
                                fp_item.source, fp_item.grp]
                    cursor.execute(sql_str, sql_args)
            return cursor.lastrowid
        except Error as e:
            print(e)
    
    
    def close(self):
        self._db_con.close()
    
    def get_grp(self, grp):
        
        if self.use_regex:
            sql_str = "SELECT * from compounds where grp REGEXP ? order by perm_order"
        else:
            sql_str = 'SELECT * from compounds where grp = ? order by perm_order'
        sql_args = [grp]
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                cursor.execute(sql_str, sql_args)
            return cursor.fetchall()
        except Error as e:
            print(e)
    
    def get_all(self):
        sql_str = 'SELECT * from compounds order by perm_order'
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                cursor.execute(sql_str)
            return cursor.fetchall()
        except Error as e:
            print(e)
            
    def get_by_mf(self, mf, grp):
        if self.use_regex:
            sql_str = 'SELECT * from compounds where mf_text = ? and grp REGEXP ? order by perm_order'
        else:
            sql_str = 'SELECT * from compounds where mf_text = ? and grp = ? order by perm_order'
        sql_args = [mf, grp]
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                cursor.execute(sql_str, sql_args)
            return cursor.fetchall()
        except Error as e:
            print(e)
            
            
    def get_count_inchikey1(self, inchikey1, grp):
        sql_str = 'SELECT count() from compounds where inchikey1 = ? and grp = ?'
        sql_args = [inchikey1, grp]
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                cursor.execute(sql_str, sql_args)
            return cursor.fetchone()["count()"]
        except Error as e:
            print(e)

            
    def delete_grp(self, grp):
        sql_str = 'DELETE from compounds where grp = ?'
        sql_args = [grp]
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                cursor.execute(sql_str, sql_args)
            return cursor.fetchall()
        except Error as e:
            print(e)
            
    def randomize(self, keep = True):
        sql_str = 'UPDATE compounds SET perm_order = random()'
        if keep:
            sql_str = 'UPDATE compounds SET perm_order = random() where perm_order IS NULL'
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                cursor.execute(sql_str)
            return cursor.fetchall()
        except Error as e:
            print(e)
            
    def sql(self, sql_str):
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                cursor.execute(sql_str)
            return cursor.fetchall()
        except Error as e:
            print(e)
        
FpDatabase.register_mapping(".db", FpDatabaseSqlite)
