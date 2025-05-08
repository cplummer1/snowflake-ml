from safety_inspections_ml.utils.snowflake.SnowflakeConnection import SnowflakeConnection
sf = SnowflakeConnection()

query = sf.load_sql_query('utils/snowflake/queries/test.sql')

print(sf.execute_query(query))
