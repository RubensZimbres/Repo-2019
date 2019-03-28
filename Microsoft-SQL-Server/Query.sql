USE MASTER;
EXEC sp_who2

KILL 72
KILL 73

USE MASTER;
CREATE DATABASE database1


SELECT VISITA_STATUS,COUNT(VISITA_STATUS) AS "Soma Consolidada" 
FROM database1.dbo.CallsDia 
LEFT JOIN database3.dbo.Diario 
ON CallsDia.Nome=Diario.NOME WHERE CallsDia.DATA_ABERTURA_SUPORTE>'2018-04-29' 
AND DATA_ABERTURA_VISITA IS NOT NULL AND VISITA_STATUS IS NOT NULL 
GROUP BY VISITA_STATUS ORDER BY "Soma Consolidada"



CREATE TABLE TabelaDados (
    ID int,
    LastName varchar(255),
    FirstName varchar(255),
    Address varchar(255),
    City varchar(255) 
)



INSERT INTO TabelaDados(ID,LastName,FirstName,Address,City)
VALUES ('128', 'da Silva', 'José','Rua do Rócio, 122', 'São Paulo');

ALTER TABLE TempDados ALTER COLUMN Timestamp date
DELETE FROM TempDados WHERE Temperature='45'
SELECT * FROM raspberry.dbo.TempDados

select suser_sname(owner_sid) as 'Owner', state_desc, *
from sys.databases


aws rds modify-db-parameter-group --db-parameter-group-name groupname --parameters 
"ParameterName='clr enabled',ParameterValue=1,ApplyMethod=immediate"

CREATE LOGIN rubens WITH PASSWORD 'rubens';
GO
EXEC sp_addsrvrolemember 'rubens', 'sysadmin';
GO

EXEC sp_configure 'external scripts enabled', 1;  
RECONFIGURE WITH OVERRIDE; 
GO

#RESTART

EXEC sp_configure  'external scripts enabled';
Go

execute sp_execute_external_script 
@language = N'Python',
@script = N'
l = [15, 18, 2, 36, 12, 78, 5, 6, 9]
print(sum(l) / float(len(l)))
'
