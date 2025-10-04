"""SQL Concepts Guide - Interactive documentation for SQL fundamentals."""

import streamlit as st
from ...utils.styling import create_section_header

def render_sql_guide():
    """Render the SQL concepts guide."""
    
    st.markdown(create_section_header("SQL Concepts & Examples"), unsafe_allow_html=True)
    
    # OLAP vs OLTP
    st.markdown("## OLAP vs OLTP Databases")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>OLTP (Online Transaction Processing)</h4>
        <ul>
        <li><strong>Purpose:</strong> Real-time transaction processing</li>
        <li><strong>Operations:</strong> INSERT, UPDATE, DELETE</li>
        <li><strong>Structure:</strong> Normalized data</li>
        <li><strong>Performance:</strong> High concurrency, low latency</li>
        <li><strong>Examples:</strong> E-commerce, banking systems</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-card">
        <h4>OLAP (Online Analytical Processing)</h4>
        <ul>
        <li><strong>Purpose:</strong> Complex analytical queries</li>
        <li><strong>Operations:</strong> SELECT with aggregations</li>
        <li><strong>Structure:</strong> Denormalized/dimensional</li>
        <li><strong>Performance:</strong> Lower concurrency, complex queries</li>
        <li><strong>Examples:</strong> Data warehouses, BI systems</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # JOIN Types
    st.markdown("## JOIN Clauses")
    
    with st.expander("All JOIN Types", expanded=False):
        st.markdown("**INNER JOIN** - Returns only matching records from both tables")
        st.markdown("*Use Case:* When you need only records that exist in both tables")
        st.code("""SELECT * FROM customers c 
INNER JOIN orders o ON c.customer_id = o.customer_id;""", language='sql')
        
        st.markdown("---")
        
        st.markdown("**LEFT JOIN** - Returns all records from left table, matched records from right")
        st.markdown("*Use Case:* When you need all customers, even those without orders")
        st.code("""SELECT * FROM customers c 
LEFT JOIN orders o ON c.customer_id = o.customer_id;""", language='sql')
        
        st.markdown("---")
        
        st.markdown("**RIGHT JOIN** - Returns all records from right table, matched records from left")
        st.markdown("*Use Case:* When you need all orders, even orphaned ones")
        st.code("""SELECT * FROM customers c 
RIGHT JOIN orders o ON c.customer_id = o.customer_id;""", language='sql')
        
        st.markdown("---")
        
        st.markdown("**FULL OUTER JOIN** - Returns all records from both tables")
        st.markdown("*Use Case:* When you need complete data from both sides")
        st.code("""SELECT * FROM customers c 
FULL OUTER JOIN orders o ON c.customer_id = o.customer_id;""", language='sql')
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # HAVING Clause
    st.markdown("## HAVING Clause")
    
    st.markdown("""
    <div class="warning-card">
    <h4>HAVING vs WHERE</h4>
    <ul>
    <li><strong>WHERE:</strong> Filters individual rows before grouping</li>
    <li><strong>HAVING:</strong> Filters groups created by GROUP BY</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.code("""
-- Example: Find departments with average salary > $50,000
-- for employees hired after 2020
SELECT 
    department, 
    AVG(salary) as avg_salary,
    COUNT(*) as employee_count
FROM employees 
WHERE hire_date > '2020-01-01'  -- Filter rows first
GROUP BY department 
HAVING AVG(salary) > 50000;     -- Then filter groups
    """, language='sql')
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Views
    st.markdown("## Database Views")
    
    st.markdown("""
    <div class="metric-card">
    <h4>What are Views?</h4>
    <p>Views are virtual tables based on SQL queries. They don't store data themselves but provide a way to present data from one or more tables.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Why Use Views Instead of Tables?**")
        st.markdown("""
        - **Security:** Hide sensitive columns
        - **Simplicity:** Simplify complex queries
        - **Reusability:** Create reusable query logic
        - **Presentation:** Format data differently
        """)
    
    with col2:
        st.code("""
CREATE VIEW high_value_customers AS
SELECT 
    customer_id, 
    name, 
    total_orders,
    total_spent
FROM customers c
JOIN (
    SELECT 
        customer_id, 
        COUNT(*) as total_orders,
        SUM(amount) as total_spent
    FROM orders 
    GROUP BY customer_id
) o ON c.customer_id = o.customer_id
WHERE total_orders > 10;
        """, language='sql')
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Primary and Foreign Keys
    st.markdown("## Primary & Foreign Keys")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="success-card">
        <h4>Primary Key</h4>
        <ul>
        <li>Uniquely identifies each row</li>
        <li>Cannot be NULL</li>
        <li>Only one per table</li>
        <li>Automatically creates unique index</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>Foreign Key</h4>
        <ul>
        <li>References primary key of another table</li>
        <li>Maintains referential integrity</li>
        <li>Can have multiple per table</li>
        <li>Can be NULL (unless specified)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.code("""
-- Example: Customer and Orders relationship
CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT,
    order_date DATE,
    amount DECIMAL(10,2),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);
    """, language='sql')
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Window Functions
    st.markdown("## Window Functions")
    
    st.markdown("""
    <div class="success-card">
    <h4>What are Window Functions?</h4>
    <p>Window functions perform calculations across a set of rows related to the current row, without collapsing the result set like GROUP BY does.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Cumulative Sum", expanded=False):
        st.markdown("**Purpose:** Calculate running totals")
        st.code("""-- Calculate cumulative revenue over time
SELECT 
    date,
    revenue,
    SUM(revenue) OVER (ORDER BY date) as cumulative_revenue,
    SUM(revenue) OVER (
        ORDER BY date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as cumulative_sum_explicit
FROM daily_sales
ORDER BY date;""", language='sql')
    
    with st.expander("LAG Function", expanded=False):
        st.markdown("**Purpose:** Access previous row values")
        st.code("""-- Compare current day with previous day
SELECT 
    date,
    revenue,
    LAG(revenue, 1) OVER (ORDER BY date) as previous_day_revenue,
    revenue - LAG(revenue, 1) OVER (ORDER BY date) as day_over_day_change,
    ROUND(
        (revenue - LAG(revenue, 1) OVER (ORDER BY date)) * 100.0 / 
        LAG(revenue, 1) OVER (ORDER BY date), 2
    ) as percent_change
FROM daily_sales
ORDER BY date;""", language='sql')
    
    with st.expander("LEAD Function", expanded=False):
        st.markdown("**Purpose:** Access next row values")
        st.code("""-- Look ahead to next employee salary in department
SELECT 
    employee_id,
    department,
    salary,
    hire_date,
    LEAD(salary, 1) OVER (
        PARTITION BY department 
        ORDER BY hire_date
    ) as next_hire_salary,
    LEAD(hire_date, 1) OVER (
        PARTITION BY department 
        ORDER BY hire_date
    ) as next_hire_date
FROM employees
ORDER BY department, hire_date;""", language='sql')
    
    with st.expander("Rolling Average", expanded=False):
        st.markdown("**Purpose:** Frame-based window calculations")
        
        st.markdown("""**Why it's called a 'rolling 5 average':**
        
        The frame includes:
        - **2 PRECEDING** rows (2 employees hired before current employee)
        - **CURRENT ROW** (the current employee) 
        - **2 FOLLOWING** rows (2 employees hired after current employee)
        
        **Total: 2 + 1 + 2 = 5 rows**""")
        
        st.markdown("**Visual Example:**")
        st.code("""Employee Order by Hire Date:
┌─────────┬──────────┬─────────────────────────┐
│ Emp ID  │ Salary   │ 5-Row Rolling Average   │
├─────────┼──────────┼─────────────────────────┤
│ A       │ 50000    │ AVG(A,B,C) = 3 rows*   │
│ B       │ 55000    │ AVG(A,B,C,D) = 4 rows* │
│ C       │ 60000    │ AVG(A,B,C,D,E) = 5 rows │ ← Full frame
│ D       │ 65000    │ AVG(B,C,D,E,F) = 5 rows │ ← Full frame  
│ E       │ 70000    │ AVG(C,D,E,F,G) = 5 rows │ ← Full frame
│ F       │ 45000    │ AVG(D,E,F,G) = 4 rows*  │
│ G       │ 80000    │ AVG(E,F,G) = 3 rows*    │
└─────────┴──────────┴─────────────────────────┘

*At the edges, fewer than 5 rows are available""")
        
        st.code("""-- Frame-based rolling average calculation
SELECT 
    employee_id,
    department,
    salary,
    hire_date,
    AVG(salary) OVER (
        PARTITION BY department 
        ORDER BY hire_date 
        ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
    ) as rolling_5_avg_salary
FROM employees
ORDER BY department, hire_date;""", language='sql')
    
    with st.expander("Combined Example", expanded=False):
        st.markdown("**Purpose:** Multiple window functions together")
        st.code("""-- Comprehensive employee analysis by department
SELECT 
    employee_id,
    department,
    salary,
    hire_date,
    -- Ranking functions
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) as salary_rank,
    DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) as dense_salary_rank,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY hire_date) as hire_sequence,
    
    -- Aggregate functions
    AVG(salary) OVER (PARTITION BY department) as dept_avg_salary,
    COUNT(*) OVER (PARTITION BY department) as dept_employee_count,
    
    -- Offset functions
    LAG(salary, 1) OVER (PARTITION BY department ORDER BY salary) as lower_salary,
    LEAD(salary, 1) OVER (PARTITION BY department ORDER BY salary) as higher_salary,
    
    -- Frame-based calculations
    AVG(salary) OVER (
        PARTITION BY department 
        ORDER BY hire_date 
        ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
    ) as rolling_5_avg_salary
FROM employees
ORDER BY department, salary DESC;""", language='sql')
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Quick Reference
    st.markdown("## Quick Reference")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h5>Query Order</h5>
        <ol>
        <li>FROM</li>
        <li>WHERE</li>
        <li>GROUP BY</li>
        <li>HAVING</li>
        <li>SELECT</li>
        <li>ORDER BY</li>
        <li>LIMIT</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="warning-card">
        <h5>Performance Tips</h5>
        <ul>
        <li>Use indexes on JOIN columns</li>
        <li>Filter early with WHERE</li>
        <li>Avoid SELECT *</li>
        <li>Use LIMIT when testing</li>
        <li>Consider partitioning for large tables</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="success-card">
        <h5>Best Practices</h5>
        <ul>
        <li>Use meaningful table aliases</li>
        <li>Qualify column names</li>
        <li>Use consistent naming</li>
        <li>Comment complex queries</li>
        <li>Test with small datasets first</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)