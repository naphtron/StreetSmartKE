# Bug Documentation

## Bug 1: `Chromadb` and `sqlite3` Version Issue

### Description
- Encountered the following runtime error:
  ```
  RuntimeError: [91mYour system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0
  ```
- This error occurs because the system's `sqlite3` version does not meet Chroma's minimum requirement of version 3.35.0 or higher.

### Solution
To resolve this issue, replace the built-in `sqlite3` module with `pysqlite3`:

```python
# Solution code
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

This workaround allows you to use a compatible `sqlite3` version by substituting it with `pysqlite3`.


## Bug 2: `langchain.vectorstores.Chroma` Deprecation

### Description
- The `langchain.vectorstores.Chroma` module is deprecated.
- Attempted to use `langchain_chroma.Chroma` as an alternative but encountered the same `sqlite3` version issue, even with `pysqlite-binary` installed.

### Solution
Switch to the updated module in the LangChain community package:

```python
# Updated import for Chroma vector store
from langchain_community.vectorstores import Chroma
```

Using `langchain_community.vectorstores.Chroma` resolves the deprecation warning and maintains compatibility with the required `sqlite3` version.

---

This documentation should help with similar issues in the future.
