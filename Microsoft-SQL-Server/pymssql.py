from os import getenv
import pymssql


server = 'database.12345.us-west-1.rds.amazonaws.com'
username = 'teste'
password = 'password'

conn = pymssql.connect(
    host=server,
    user=username,
    password=password,
    database='databaseXX'
)


for i in range(0,df.shape[0]):
    cursor = conn.cursor()

    cursor.execute("INSERT INTO TabelaDados VALUES ({})".format(",".join(repr(e) for e in np.array(df.iloc[i,:]))))

    conn.commit()
