import sqlite3
import numpy as np
from sqlite3 import Error
from pandas import DataFrame as df
import re

from mpunet.database import default_tables


class DBConnection(object):
    def __init__(self, db_file_path):
        self.db_file_path = db_file_path
        self._connection = None
        self._cursor = None
        self._connected = False

        # Boolean, return results from query directly
        self._echo = True
        self._auto_commit = True

        # Dict pointing to default table objects
        self.default_tables = {}
        self.create_default_table("DefaultTablesTable")

    def __repr__(self):
        """
        Developer string representation

        :return: str
        """
        return "DBConnection(%s)" % self.db_file_path

    def __str__(self):
        """
        Client string representation

        :return: str
        """
        s = "Database connection\n"
        s += "-"*len(s) + "\n"
        s += "DB:     %s\n" % self.db_file_path +\
             "Status: %s" % ("Connected" if self.connected else "Closed")
        return s

    def __call__(self, *args, **kwargs):
        """
        Delegates to self.query

        :param args: any arguments
        :param kwargs: any keyword arguments
        :return: return value of query method
        """
        if not self.connected:
            with self as conn:
                response = self.query(*args, **kwargs)
        else:
            response = self.query(*args, **kwargs)
        return response

    @property
    def echo(self):
        """
        Echo result set from query property getter

        :return: Boolean
        """
        return self._echo

    @echo.setter
    def echo(self, value):
        """
        Echo result set from query property setter

        :param value: boolean
        :return: None
        """
        if not isinstance(value, bool):
            raise ValueError("Echo must be True or False")
        self._echo = value

    @property
    def auto_commit(self):
        """
        Automatically commit all actions

        :return: Boolean
        """
        return self._auto_commit

    @auto_commit.setter
    def auto_commit(self, value):
        """
        Automatically commit all actions

        :param value: boolean
        :return: None
        """
        if not isinstance(value, bool):
            raise ValueError("Auto commit must be True or False")
        self._auto_commit = value

    def query(self, q_string, echo=None, to_numpy=True):
        """
        Executes a SQL query string against the database.
        Echoes the result back if echo=True or self.echo=True.

        :param q_string: SQL query string
        :param echo: echo back result, overwrites self.echo
        :return: SQL result set or Cursor object
        """
        words = q_string.upper().split(" ")
        if "DROP" in words:
            raise UserWarning("DROP queries should be performed manually.")

        # Define cursor object
        try:
            self.cursor = self.connection.cursor()
        except AttributeError:
            raise Error("Connection to DB has not been established.")

        # Execute query
        try:
            self.cursor.execute(q_string)
            if self.auto_commit:
                self.connection.commit()
        except Error as e:
            err_str = 'Error in query:\n"""\n%s\n"""' % q_string
            raise Error(err_str) from e

        # Fetch or return cursor
        echo = self.echo if echo is None else echo
        if echo and "SELECT" in words:
            if to_numpy:
                return np.array(self.cursor.fetchall())
            else:
                return self.cursor.fetchall()

    @property
    def connected(self):
        """
        Property getter for connection status (True/False)

        :return: boolean
        """
        return self._connected

    @property
    def connection(self):
        """
        Getter property for SQLite connection object

        :return: sqlite3.Connection or None
        """
        return self._connection

    @property
    def cursor(self):
        """
        Property getter for DB Cursor object

        :return: sqlite3.Cursor object
        """
        return self._cursor

    @cursor.setter
    def cursor(self, value):
        """
        Property setter for DB Cursor object

        :param value: sqlite.Cursor object
        :return: None
        """
        if not isinstance(value, sqlite3.Cursor):
            raise ValueError("cursor must be a sqlite3 Cursor object.")
        self._cursor = value

    @property
    def tables(self):
        """
        Return the names of all tables in the DB

        :return: list of strings of table names
        """
        with self as db:
            names = db.query("SELECT name FROM sqlite_master WHERE type='table';",
                             echo=True)
        return [n[0] for n in names]

    def print_table_info(self, table_name):
        """
        Return a list of column names for the table

        :param table_name: str, name of the table to fetch info on
        :return: list
        """
        if table_name not in self.tables:
            raise ValueError("No table '%s' in the DB." % table_name)

        tab_info = self("SELECT sql FROM sqlite_master "
                        "WHERE tbl_name = '%s' "
                        "AND type = 'table'" % table_name,
                        echo=True, to_numpy=False)[0][0].replace("\n", "")

        # Parse the info slightly
        columns = tab_info.split(f"{table_name}")[-1].strip(" (").split(",")
        print(f"Table name: {table_name}")
        for c in columns:
            name, type_ = c.split(" ", 1)
            print(name.ljust(25), type_)

    def create_default_table(self, default_table_name, **kwargs):
        # Create table object
        table = default_tables.__dict__[default_table_name](**kwargs)

        if default_table_name in self.tables:
            if default_table_name in self.default_tables:
                print("[OBS] Default table '%s' already exists" % default_table_name)
                return
        else:
            self(table.get_create_query(**kwargs))

        # Insert into dictionary
        self.default_tables[default_table_name] = table

        # For persistence, insert the table information into the
        # DefaultTablesTable from which the table object can be recreated at
        # a later time
        self.insert_into_default("DefaultTablesTable",
                                 table_name=default_table_name,
                                 **kwargs)

    def insert_into_default(self, default_table_name, **kwargs):
        # Get table and insert
        try:
            table = self.default_tables[default_table_name]
        except KeyError:
            # Recreate the table object
            table = None
        self(table.get_insert_query(**kwargs))

    def connect(self):
        """
        Attempts to connect to the SQLite database

        :return: None
        """
        try:
            self._connection = sqlite3.connect(self.db_file_path)
            self._connected = True

            # Enable foreign key support
            self.query("PRAGMA foreign_keys = ON;")

        except Error as e:
            raise Error("Could not connect to: %s" % self.db_file_path) from e
        return self

    def disconnect(self):
        """
        Disconnects from the SQLite database

        :return: None
        """
        self.connection.close()
        self._connected = False

    def __enter__(self):
        """
        On enter method for context manager usage. Establishes connection.

        :return: Connected DBConnection object
        """
        self.connect()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """
        On leave method for context manager usage. Disconnects the connection.
        Arguments are passed by Python automatically if an exception is raised
        within the context management block.

        :param exception_type: type of exception raised within context.
        :param exception_value: exception value from raised exception
        :param traceback: traceback of raised exception
        :return: None
        """
        self.disconnect()
