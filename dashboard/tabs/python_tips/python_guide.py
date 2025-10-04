"""Python Tips Guide - Programming best practices and Python-specific concepts."""

import streamlit as st
from ...utils.styling import create_section_header

def render_python_guide():
    """Render the Python tips guide."""
    
    st.markdown(create_section_header("Python Tips & Programming Best Practices"), unsafe_allow_html=True)
    
    # Programming Best Practices
    st.markdown("## Programming Best Practices")
    
    with st.expander("SOLID Principles", expanded=False):
        st.markdown("""
        **S - Single Responsibility Principle**
        - A class should have only one reason to change
        - Each class should have one job or responsibility
        
        **O - Open/Closed Principle**
        - Software entities should be open for extension but closed for modification
        - Use inheritance and polymorphism instead of modifying existing code
        
        **L - Liskov Substitution Principle**
        - Objects of a superclass should be replaceable with objects of its subclasses
        - Derived classes must be substitutable for their base classes
        
        **I - Interface Segregation Principle**
        - Many client-specific interfaces are better than one general-purpose interface
        - Don't force classes to implement interfaces they don't use
        
        **D - Dependency Inversion Principle**
        - Depend on abstractions, not concretions
        - High-level modules should not depend on low-level modules
        """)
        
        st.code("""
# Example: Single Responsibility Principle
# Bad - Multiple responsibilities
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
    
    def save_to_database(self):
        # Database logic here
        pass
    
    def send_email(self):
        # Email logic here
        pass

# Good - Single responsibility
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email

class UserRepository:
    def save(self, user):
        # Database logic here
        pass

class EmailService:
    def send_email(self, user):
        # Email logic here
        pass
        """, language='python')
    
    with st.expander("DRY and Other Principles", expanded=False):
        st.markdown("""
        **DRY - Don't Repeat Yourself**
        - Every piece of knowledge must have a single, unambiguous representation
        - Avoid code duplication by extracting common functionality
        
        **KISS - Keep It Simple, Stupid**
        - Simple solutions are better than complex ones
        - Avoid unnecessary complexity
        
        **YAGNI - You Aren't Gonna Need It**
        - Don't implement functionality until you actually need it
        - Avoid over-engineering
        
        **Boy Scout Rule**
        - Always leave the code cleaner than you found it
        - Make small improvements whenever you touch code
        """)
        
        st.code("""
# DRY Example
# Bad - Repetitive code
def calculate_circle_area(radius):
    return 3.14159 * radius * radius

def calculate_circle_circumference(radius):
    return 2 * 3.14159 * radius

# Good - Extract common values
import math

def calculate_circle_area(radius):
    return math.pi * radius ** 2

def calculate_circle_circumference(radius):
    return 2 * math.pi * radius
        """, language='python')
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # What Good Code Looks Like
    st.markdown("## What Good Code Looks Like")
    
    st.markdown("""
    <div class="success-card">
    <h4>Characteristics of Good Code</h4>
    <ul>
    <li><strong>Readable:</strong> Clear variable names, proper formatting, logical structure</li>
    <li><strong>Maintainable:</strong> Easy to modify and extend without breaking existing functionality</li>
    <li><strong>Testable:</strong> Written in a way that makes unit testing straightforward</li>
    <li><strong>Modular:</strong> Broken into small, focused functions and classes</li>
    <li><strong>Consistent:</strong> Follows established coding conventions and patterns</li>
    <li><strong>Documented:</strong> Has clear docstrings and comments where necessary</li>
    <li><strong>Efficient:</strong> Performs well without premature optimization</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.code("""
# Good code example
def calculate_customer_lifetime_value(
    monthly_revenue: float, 
    retention_months: int, 
    discount_rate: float = 0.1
) -> float:
    '''
    Calculate customer lifetime value using discounted cash flow.
    
    Args:
        monthly_revenue: Average monthly revenue per customer
        retention_months: Expected customer retention period
        discount_rate: Monthly discount rate for NPV calculation
    
    Returns:
        Customer lifetime value in dollars
    '''
    if monthly_revenue <= 0 or retention_months <= 0:
        raise ValueError("Revenue and retention must be positive")
    
    total_value = 0
    for month in range(1, retention_months + 1):
        discounted_revenue = monthly_revenue / ((1 + discount_rate) ** month)
        total_value += discounted_revenue
    
    return round(total_value, 2)
    """, language='python')
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Composition over Inheritance
    st.markdown("## Composition over Inheritance")
    
    with st.expander("Composition vs Inheritance", expanded=False):
        st.markdown("""
        **Inheritance ("is-a" relationship)**
        - Creates tight coupling between parent and child classes
        - Changes in parent class can break child classes
        - Limited flexibility - single inheritance in Python
        
        **Composition ("has-a" relationship)**
        - Loose coupling between classes
        - More flexible and maintainable
        - Easier to test and modify
        - Allows multiple behaviors to be combined
        """)
        
        st.code("""
# Inheritance approach (less flexible)
class Animal:
    def move(self):
        pass

class Bird(Animal):
    def move(self):
        return "Flying"
    
    def make_sound(self):
        return "Tweet"

class Penguin(Bird):  # Problem: Penguins can't fly!
    def move(self):
        return "Swimming"  # Overriding inherited behavior

# Composition approach (more flexible)
class FlyingBehavior:
    def move(self):
        return "Flying"

class SwimmingBehavior:
    def move(self):
        return "Swimming"

class SoundBehavior:
    def make_sound(self):
        return "Tweet"

class Bird:
    def __init__(self, movement_behavior, sound_behavior):
        self.movement = movement_behavior
        self.sound = sound_behavior
    
    def move(self):
        return self.movement.move()
    
    def make_sound(self):
        return self.sound.make_sound()

# Usage
eagle = Bird(FlyingBehavior(), SoundBehavior())
penguin = Bird(SwimmingBehavior(), SoundBehavior())
        """, language='python')
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Python-Specific Concepts
    st.markdown("## Python-Specific Concepts")
    
    with st.expander("Decorators", expanded=False):
        st.markdown("""
        **What are Decorators?**
        - Functions that modify or enhance other functions without changing their code
        - Used for logging, timing, authentication, caching, etc.
        - Implemented using the `@decorator_name` syntax
        """)
        
        st.code("""
import time
from functools import wraps

# Simple decorator
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Usage
@timer
def slow_function():
    time.sleep(1)
    return "Done"

# Decorator with parameters
def retry(max_attempts=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed, retrying...")
            return None
        return wrapper
    return decorator

@retry(max_attempts=3)
def unreliable_api_call():
    # Simulate API call that might fail
    import random
    if random.random() < 0.7:
        raise Exception("API call failed")
    return "Success"
        """, language='python')
    
    with st.expander("Iterators and Generators", expanded=False):
        st.markdown("""
        **Iterators**
        - Objects that implement `__iter__()` and `__next__()` methods
        - Can be used in for loops and with `next()` function
        - Remember their state between calls
        
        **Generators**
        - Special type of iterator created using `yield` keyword
        - Memory efficient - generate values on-demand
        - Automatically implement iterator protocol
        """)
        
        st.code("""
# Iterator example
class NumberIterator:
    def __init__(self, max_num):
        self.max_num = max_num
        self.current = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current < self.max_num:
            self.current += 1
            return self.current
        raise StopIteration

# Generator function (much simpler)
def number_generator(max_num):
    current = 0
    while current < max_num:
        current += 1
        yield current

# Generator expression (even simpler)
squares = (x**2 for x in range(10))

# Usage
for num in number_generator(5):
    print(num)  # Prints 1, 2, 3, 4, 5

# Memory efficient file processing
def read_large_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

# Process file without loading entire file into memory
for line in read_large_file('huge_file.txt'):
    process_line(line)
        """, language='python')
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Python Data Structures
    st.markdown("## Python Data Structures")
    
    with st.expander("List vs Tuple", expanded=False):
        st.markdown("""
        **Lists**
        - Mutable (can be changed after creation)
        - Ordered collection of items
        - Use square brackets `[]`
        - Slower for iteration (due to mutability overhead)
        
        **Tuples**
        - Immutable (cannot be changed after creation)
        - Ordered collection of items
        - Use parentheses `()`
        - Faster for iteration
        - Can be used as dictionary keys (hashable)
        """)
        
        st.code("""
# Lists - Mutable
my_list = [1, 2, 3, 4]
my_list.append(5)        # [1, 2, 3, 4, 5]
my_list[0] = 10          # [10, 2, 3, 4, 5]
my_list.extend([6, 7])   # [10, 2, 3, 4, 5, 6, 7]

# Tuples - Immutable
my_tuple = (1, 2, 3, 4)
# my_tuple.append(5)     # Error! Tuples don't have append
# my_tuple[0] = 10       # Error! Cannot modify tuple

# Use cases
coordinates = (10.5, 20.3)  # Immutable point
rgb_color = (255, 128, 0)   # Color values shouldn't change

# Tuple as dictionary key
locations = {
    (0, 0): "Origin",
    (10, 20): "Point A",
    (30, 40): "Point B"
}

# Converting between list and tuple
list_to_tuple = tuple([1, 2, 3])    # (1, 2, 3)
tuple_to_list = list((1, 2, 3))     # [1, 2, 3]
        """, language='python')
    
    with st.expander("Append vs Extend", expanded=False):
        st.markdown("""
        **append()**
        - Adds a single element to the end of the list
        - The element is added as-is (even if it's a list)
        - Returns None (modifies original list)
        
        **extend()**
        - Adds multiple elements from an iterable to the end of the list
        - Unpacks the iterable and adds each element individually
        - Returns None (modifies original list)
        """)
        
        st.code("""
# append() - adds single element
my_list = [1, 2, 3]
my_list.append(4)           # [1, 2, 3, 4]
my_list.append([5, 6])      # [1, 2, 3, 4, [5, 6]] - list as single element

# extend() - adds multiple elements
my_list = [1, 2, 3]
my_list.extend([4, 5, 6])   # [1, 2, 3, 4, 5, 6]
my_list.extend("abc")       # [1, 2, 3, 4, 5, 6, 'a', 'b', 'c']

# Alternative to extend using += operator
my_list = [1, 2, 3]
my_list += [4, 5, 6]        # [1, 2, 3, 4, 5, 6] - same as extend

# Performance comparison for adding multiple elements
import time

# Slow - multiple append calls
my_list = []
for item in range(1000):
    my_list.append(item)

# Fast - single extend call
my_list = []
my_list.extend(range(1000))
        """, language='python')
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Memory Management
    st.markdown("## Memory Management in Python")
    
    with st.expander("Python Memory Management", expanded=False):
        st.markdown("""
        **Reference Counting**
        - Python tracks how many references point to each object
        - When reference count reaches zero, memory is freed
        - Handles most memory management automatically
        
        **Garbage Collection**
        - Handles circular references that reference counting can't
        - Runs periodically to clean up unreachable objects
        - Can be controlled with the `gc` module
        
        **Memory Pools**
        - Python uses memory pools for small objects
        - Reduces fragmentation and improves performance
        - Objects under 512 bytes use specialized allocators
        """)
        
        st.code("""
import sys
import gc

# Reference counting example
a = [1, 2, 3]
print(sys.getrefcount(a))  # Reference count (usually 2: 'a' and getrefcount parameter)

b = a  # Increase reference count
print(sys.getrefcount(a))  # Now higher

del b  # Decrease reference count
print(sys.getrefcount(a))  # Back to original

# Circular reference example
class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.parent = None

# Create circular reference
parent = Node("parent")
child = Node("child")
parent.children.append(child)
child.parent = parent  # Circular reference

# Even after deleting variables, objects might not be freed immediately
del parent, child

# Force garbage collection
collected = gc.collect()
print(f"Collected {collected} objects")

# Memory optimization tips
# 1. Use generators for large datasets
def process_large_dataset():
    for item in range(1000000):
        yield process_item(item)  # Memory efficient

# 2. Use __slots__ for classes with many instances
class Point:
    __slots__ = ['x', 'y']  # Reduces memory overhead
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        """, language='python')
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # DataFrames and Spark
    st.markdown("## DataFrames and Spark")
    
    with st.expander("What is a DataFrame?", expanded=False):
        st.markdown("""
        **DataFrame**
        - 2-dimensional labeled data structure (like a table or spreadsheet)
        - Columns can contain different data types
        - Primary data structure in pandas for data analysis
        - Built on top of numpy arrays for performance
        - Provides powerful data manipulation and analysis tools
        """)
        
        st.code("""
import pandas as pd
import numpy as np

# Creating DataFrames
# From dictionary
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 70000]
}
df = pd.DataFrame(data)

# From CSV file
# df = pd.read_csv('data.csv')

# Basic operations
print(df.head())           # First 5 rows
print(df.info())           # Data types and memory usage
print(df.describe())       # Statistical summary

# Data manipulation
df['bonus'] = df['salary'] * 0.1          # Add new column
high_earners = df[df['salary'] > 55000]   # Filter rows
grouped = df.groupby('department').mean() # Group by operations

# Data cleaning
df.dropna()                # Remove missing values
df.fillna(0)              # Fill missing values
df.drop_duplicates()      # Remove duplicates

# Merging DataFrames
df1 = pd.DataFrame({'id': [1, 2], 'name': ['A', 'B']})
df2 = pd.DataFrame({'id': [1, 2], 'score': [85, 90]})
merged = pd.merge(df1, df2, on='id')
        """, language='python')
    
    with st.expander("Spark: DataFrame vs RDD", expanded=False):
        st.markdown("""
        **RDD (Resilient Distributed Dataset)**
        - Low-level abstraction in Spark
        - Immutable distributed collection of objects
        - Functional programming interface
        - No built-in optimization
        - More flexible but requires more coding
        
        **DataFrame (Spark SQL)**
        - Higher-level abstraction built on top of RDDs
        - Structured data with schema (like database table)
        - Catalyst optimizer for query optimization
        - Easier to use with SQL-like operations
        - Better performance due to optimizations
        """)
        
        st.code("""
from pyspark.sql import SparkSession
from pyspark import SparkContext

spark = SparkSession.builder.appName("DataFrameVsRDD").getOrCreate()
sc = spark.sparkContext

# RDD Example - Lower level, more manual
rdd_data = [("Alice", 25, 50000), ("Bob", 30, 60000), ("Charlie", 35, 70000)]
rdd = sc.parallelize(rdd_data)

# RDD operations (functional style)
filtered_rdd = rdd.filter(lambda x: x[2] > 55000)  # Filter by salary
mapped_rdd = rdd.map(lambda x: (x[0], x[2] * 1.1))  # Name and salary with 10% raise
result_rdd = filtered_rdd.collect()

# DataFrame Example - Higher level, SQL-like
df_data = [("Alice", 25, 50000), ("Bob", 30, 60000), ("Charlie", 35, 70000)]
columns = ["name", "age", "salary"]
df = spark.createDataFrame(df_data, columns)

# DataFrame operations (SQL-like)
df.filter(df.salary > 55000).show()  # Filter by salary
df.select("name", (df.salary * 1.1).alias("new_salary")).show()  # Select with calculation

# SQL queries on DataFrames
df.createOrReplaceTempView("employees")
spark.sql(\"\"\"
    SELECT name, salary * 1.1 as new_salary 
    FROM employees 
    WHERE salary > 55000
\"\"\").show()

# Performance differences
# DataFrame: Uses Catalyst optimizer, columnar storage, code generation
# RDD: No automatic optimization, row-based processing

# When to use each:
# Use DataFrame for:
# - Structured data analysis
# - SQL-like operations
# - Better performance with large datasets
# - Integration with Spark SQL

# Use RDD for:
# - Complex data transformations
# - Working with unstructured data
# - Need fine-grained control over data processing
# - Custom partitioning logic
        """, language='python')
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Quick Reference
    st.markdown("## Quick Reference")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h5>Code Quality Checklist</h5>
        <ul>
        <li>Clear, descriptive names</li>
        <li>Functions do one thing</li>
        <li>Consistent formatting</li>
        <li>Proper error handling</li>
        <li>Unit tests included</li>
        <li>Documentation present</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-card">
        <h5>Python Performance Tips</h5>
        <ul>
        <li>Use list comprehensions</li>
        <li>Leverage built-in functions</li>
        <li>Use generators for large data</li>
        <li>Profile before optimizing</li>
        <li>Consider numpy for numerics</li>
        <li>Use appropriate data structures</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="success-card">
        <h5>Design Patterns</h5>
        <ul>
        <li>Favor composition over inheritance</li>
        <li>Use dependency injection</li>
        <li>Apply SOLID principles</li>
        <li>Keep interfaces simple</li>
        <li>Write testable code</li>
        <li>Follow established conventions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)