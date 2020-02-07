import pickle
import sqlite3


class DefaultTablesTable(object):

    def __init__(self):
        self.table_name = self.__class__.__name__

    def get_create_query(self, **kwargs):
        return f"CREATE TABLE {self.table_name} (id INTEGER PRIMARY KEY," \
               f"table_name VARCHAR(255),kwargs BLOB);"

    def get_insert_query(self, table_name, **kwargs):
        bdata = pickle.dumps(kwargs, pickle.HIGHEST_PROTOCOL)
        return f"INSERT INTO {self.table_name} (table_name,kwargs) VALUES " \
               f"({table_name}, {sqlite3.Binary(bdata)});"


class ResultsByView(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.table_name = self.__class__.__name__

    def get_create_query(self, **kwargs):
        query = f"CREATE TABLE {self.table_name} (id INTEGER PRIMARY KEY," \
                f"image_id VARCHAR(255) NOT NULL,mean_dice FLOAT(5),"

        for i in range(self.n_classes):
            query += f"class_{i}_dice FLOAT(5),"
        return query[:-1] + ");"

    def get_insert_query(self, image_id, mean_dice, per_class_dices):
        query = f"INSERT INTO {self.table_name} (image_id,mean_dice,"
        for i in range(self.n_classes):
            query += f"class_{i}_dice,"
        query = query[:-1]
        query += f") VALUES ('{image_id}',{mean_dice},"
        for i in range(self.n_classes):
            query += f"{per_class_dices[i]},"
        return query[:-1] + ");"
