import sqlite3


class DataBase:
    """DataBase class holds a database instance.

    Attributes:
        cursor: used for executing queries.
    """

    def __init__(self, db_file_name: str):
        """Initializes the database instance.
        Args:
          db_file_name: the name of the database file.
        """
        conn = sqlite3.connect(db_file_name)
        self.cursor = conn.cursor()

    def execute(self, query: str):
        """Executes one query on the database.
        Args:
          query: tstring query to execute.
        Returns: the rows returned by the query.
        """
        return self.cursor.execute(query).fetchall()
