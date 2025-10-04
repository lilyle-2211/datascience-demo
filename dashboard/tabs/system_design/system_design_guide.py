"""System Design Guide - Scalable architecture patterns and design principles."""

import streamlit as st
from ...utils.styling import create_section_header

def render_system_design_guide():
    """Render the System Design guide."""
    
    st.markdown(create_section_header("System Design Principles & Patterns"), unsafe_allow_html=True)
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Security & Monitoring
    st.markdown("## Security & Monitoring")
    
    with st.expander("Security Patterns", expanded=False):
        st.markdown("""
        **Authentication & Authorization**
        - OAuth 2.0 / OpenID Connect for user authentication
        - JWT tokens for stateless authentication
        - Role-based access control (RBAC)
        - API keys and rate limiting
        
        **Data Security**
        - Encryption at rest and in transit
        - Data masking and tokenization
        - Audit logging and compliance
        - Network security (VPCs, firewalls)
        """)
        
        st.code("""
# Simple Authentication
class SimpleAuth:
    def __init__(self):
        self.sessions = {}  # In production, use Redis/database
    
    def login(self, username, password):
        # Check credentials (simplified)
        if username == "admin" and password == "secret":
            session_id = f"session_{username}_123"
            self.sessions[session_id] = {"user": username, "role": "admin"}
            return session_id
        return None
    
    def check_auth(self, session_id):
        return self.sessions.get(session_id)

# Rate Limiting
class RateLimiter:
    def __init__(self, max_requests=100):
        self.max_requests = max_requests
        self.requests = {}
    
    def is_allowed(self, client_ip):
        count = self.requests.get(client_ip, 0)
        if count < self.max_requests:
            self.requests[client_ip] = count + 1
            return True
        return False

# Usage
auth = SimpleAuth()
limiter = RateLimiter(max_requests=10)

# Check if request is allowed
if limiter.is_allowed("192.168.1.1"):
    # Process request
    pass
else:
    # Return rate limit error
    pass
        """, language='python')
    
    with st.expander("Monitoring & Observability", expanded=False):
        st.markdown("""
        **The Three Pillars of Observability**
        - **Metrics:** Numerical measurements over time (CPU, memory, request count)
        - **Logs:** Discrete events with timestamps and context
        - **Traces:** Request flow through distributed systems
        
        **Key Metrics to Monitor**
        - System metrics: CPU, memory, disk, network
        - Application metrics: Response time, throughput, error rate
        - Business metrics: User engagement, conversion rates
        """)
        
        st.code("""
# Simple Monitoring
import time
import logging

class SimpleMonitor:
    def __init__(self):
        self.metrics = {"requests": 0, "errors": 0}
        
    def record_request(self):
        self.metrics["requests"] += 1
    
    def record_error(self):
        self.metrics["errors"] += 1
    
    def get_stats(self):
        return self.metrics

# Basic Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_request(method, url, status, duration):
    logger.info(f"{method} {url} - {status} - {duration}ms")

# Health Check
def health_check():
    try:
        # Check database connection
        database.ping()
        return {"status": "healthy", "database": "ok"}
    except:
        return {"status": "unhealthy", "database": "error"}

# Usage
monitor = SimpleMonitor()
monitor.record_request()
stats = monitor.get_stats()
print(f"Total requests: {stats['requests']}")
        """, language='python')
    
