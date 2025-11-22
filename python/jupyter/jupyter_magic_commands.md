The `%` and `%%` symbols are not part of the Python language itself, but are special commands 
for **IPython** and **Jupyter** environments.

They are called **magic commands**, and they add powerful, interactive features to your workflow.

### 1. The Single `%` (Line Magic)

A single `%` prefix applies a command to a **single line** of code.

#### Example: `%timeit`
The `%timeit` command is used for **performance testing**. It automatically runs a piece of code multiple times 
to get a reliable estimate of its execution time.

**Other common line magics:**
*   `%run`: Execute a Python script.
*   `%load`: Insert code from an external file into a cell.
*   `%who`: List all variables in the namespace.
*   `%matplotlib inline`: Display matplotlib plots inline in the notebook.

### 2. The Double `%%` (Cell Magic)

A double `%%` prefix applies a command to the **entire cell**. It controls how the entire content of the cell is processed.

#### Example: `%%cython -a`
The `%%cython` magic is used to **compile and run Cython code** directly in a cell. Cython is a superset of Python 
that compiles to C for massive performance gains.

The `-a` flag (short for `--annotate`) is particularly important. It generates an **annotated HTML report** that 
shows which lines in your Cython code are interacting with Python (which is slow) and which are pure, fast C.

This is an incredibly powerful tool for high-performance computing directly within the interactive notebook environment.

**Other common cell magics:**
*   `%%timeit`: Time the execution of the entire cell.
*   `%%writefile`: Write the cell's content to a file.
*   `%%bash`: Run the cell's content in a Bash shell.
*   `%%html`: Render the cell's content as HTML.

### Summary Table

| Symbol   | Name           | Scope          | Purpose                                                        | Key Examples                             |
|:---------|:---------------|:---------------|:---------------------------------------------------------------|:-----------------------------------------|
| **`%`**  | **Line Magic** | A single line  | Perform useful, interactive tasks on a line of code.           | `%timeit`, `%run`, `%who`                |
| **`%%`** | **Cell Magic** | An entire cell | Process or execute the entire cell's content in a special way. | `%%cython -a`, `%%timeit`, `%%writefile` |

### How to See All Magics

You can list all available magic commands in your IPython/Jupyter session by typing:

```python
%lsmagic
```
And you can get help on any specific magic by using `?`:
```python
%timeit?
```
=====================================================================================================

While both `%load` and `import` deal with code reuse, they serve **fundamentally different purposes** 
and are used at **different stages** of your workflow.

## `%load` is for **Development/Exploration**
## `import` is for **Execution/Production**

**When you use `%load`:**
```python
# This loads the entire content of my_script.py INTO the current cell
%load my_script.py
```

**After running this, your cell content BECOMES:**
```python
# Content of my_script.py:
def calculate_stats(data):
    return sum(data) / len(data)

numbers = [1, 2, 3, 4, 5]
result = calculate_stats(numbers)
print(f"Average: {result}")
```

**Use Cases for `%load`:**
1. **Debugging & Exploration**: You want to see, modify, and experiment with someone else's code
2. **Learning**: You're studying how a function works and want to step through it line by line
3. **Code Review**: You need to examine and potentially modify code before incorporating it
4. **Prototyping**: You want to build upon existing code by copying and adapting it

**Key Point**: `%load` gives you the **source code** to work with directly.

### `import` - The "Use Without Seeing" Approach

**When you use `import`:**
```python
import my_script
# OR
from my_script import calculate_stats

# Now you can USE the function, but you don't SEE its implementation
result = calculate_stats([1, 2, 3, 4, 5])
print(result)  # Output: 3.0
```

**Use Cases for `import`:**
1. **Using Stable Code**: The module contains tested, reliable functions you want to use
2. **Code Organization**: You've structured your project into separate, reusable modules
3. **Production Code**: You're writing a script that depends on other well-defined components
4. **Black Box Usage**: You don't need to see the implementation details, just the interface

**Key Point**: `import` gives you access to the **functionality** without exposing the implementation.

## Practical Scenarios

### Scenario 1: Learning from a Colleague's Code
```python
# You want to understand how Sarah's analysis works
%load sarahs_analysis.py
# Now you can see all her functions, modify parameters, and learn
```

### Scenario 2: Using a Well-Tested Library
```python
# You just need the functionality
import numpy as np
import pandas as pd
# Use them without seeing thousands of lines of source code
```

### Scenario 3: Debugging a Function
```python
# A function from utils.py is behaving strangely
%load utils.py
# Now you can add print statements, modify it, and see what's wrong
```

### Scenario 4: Building an Application
```python
# In your main application file
from database import connect_db
from analysis import run_calculations
from reporting import generate_report
# Clean, modular code that uses other components
```

## Key Differences Table

| Aspect           | `%load`                              | `import`                            |
|------------------|--------------------------------------|-------------------------------------|
| **Purpose**      | Code inspection, learning, debugging | Code execution, modular programming |
| **What you get** | Source code text                     | Access to functions/classes         |
| **Modification** | You can modify the loaded code       | You use the code as-is              |
| **Typical Use**  | Development, exploration             | Production, stable code reuse       |
| **Scope**        | Usually temporary                    | Permanent dependency                |
| **Performance**  | Loads once, then runs                | Imports every time script runs      |

