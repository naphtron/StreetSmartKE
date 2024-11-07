# Bug 1: Chromadb and Sqlite issue
## Description
- RuntimeError: [91mYour system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0 
## Solution
- Paste
- __import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Bug 2: langchain.vectorstores.Chroma
## Description
- Deprecated
- Used langchain_chroma.Chroma instead; recreated the ChromaDB, Sqlite issue despite using pysqlite-binary
## Solution
- use langchain_community.vectorstores.Chroma
