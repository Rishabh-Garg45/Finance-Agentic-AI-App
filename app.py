import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq 
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .agent-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False
    st.session_state.websearch_agent = None
    st.session_state.finance_agent = None
    st.session_state.multi_ai_agents = None

def initialize_agents():
    """Initialize the AI agents"""
    try:
        # Check if Groq API key is available
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
            return False
        
        # Web search agent
        websearch_agent = Agent(
            name="Web Search Agent",
            role="Search the web for the information",
            model=Groq(id="llama3-70b-8192"),
            tools=[DuckDuckGo()],
            instructions=["Always include sources"],
            show_tool_calls=True,
            markdown=True,
        )
        
        # Finance agent
        finance_agent = Agent(
            name="Finance AI Agent",
            model=Groq(id="llama3-70b-8192"),
            tools=[YFinanceTools(
                stock_price=True, 
                analyst_recommendations=True, 
                stock_fundamentals=True, 
                company_news=True
            )],
            instructions=["Use tables to display the data"],
            show_tool_calls=True,
            markdown=True,
        )
        
        # Multi-agent system
        multi_ai_agents = Agent(
            model=Groq(id="llama3-70b-8192"),
            team=[websearch_agent, finance_agent],
            instructions=["Always include instructions", "Use table to display data"],
            show_tool_calls=True,
            markdown=True,
        )
        
        # Store in session state
        st.session_state.websearch_agent = websearch_agent
        st.session_state.finance_agent = finance_agent
        st.session_state.multi_ai_agents = multi_ai_agents
        st.session_state.agents_initialized = True
        
        return True
        
    except Exception as e:
        st.error(f"Error initializing agents: {str(e)}")
        return False

def clean_response(response_text):
    """Clean the response text to remove unwanted characters"""
    # Remove ANSI escape codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned = ansi_escape.sub('', response_text)
    return cleaned

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 Multi-Agent Stock Analysis Platform</h1>
        <p>Comprehensive stock analysis powered by AI agents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 AI Agents")
        
        st.markdown("""
        <div class="agent-card">
            <h4>🔍 Web Search Agent</h4>
            <p>Searches the web for latest information and news</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="agent-card">
            <h4>💰 Finance Agent</h4>
            <p>Analyzes stock data, fundamentals, and analyst recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="agent-card">
            <h4>🎯 Multi-Agent System</h4>
            <p>Coordinates both agents for comprehensive analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📋 Requirements")
        st.markdown("""
        - GROQ_API_KEY in .env file
        - OPENAI_API_KEY in .env file
        - Internet connection for web search
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Stock Analysis")
        
        # Stock symbol input
        stock_symbol = st.text_input(
            "Enter Stock Symbol (e.g., NVDA, AAPL, TSLA)",
            value="NVDA",
            help="Enter a valid stock ticker symbol"
        ).upper()
        
        # Analysis options
        analysis_type = st.selectbox(
            "Select Analysis Type",
            [
                "Complete Analysis (Recommendations + News)",
                "Analyst Recommendations Only",
                "Latest News Only",
                "Stock Fundamentals",
                "Custom Query"
            ]
        )
        
        # Custom query input
        custom_query = ""
        if analysis_type == "Custom Query":
            custom_query = st.text_area(
                "Enter your custom query",
                placeholder="e.g., Compare NVDA with AMD in terms of AI market position",
                height=100
            )
    
    with col2:
        st.header("Quick Actions")
        
        # Popular stocks
        st.subheader("Popular Stocks")
        popular_stocks = ["NVDA", "AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]
        
        for stock in popular_stocks:
            if st.button(f"Analyze {stock}", key=f"quick_{stock}"):
                st.session_state.selected_stock = stock
                st.rerun()
        
        # Check if a quick stock was selected
        if hasattr(st.session_state, 'selected_stock'):
            stock_symbol = st.session_state.selected_stock
            delattr(st.session_state, 'selected_stock')
    
    # Analysis execution
    if st.button("🚀 Run Analysis", type="primary"):
        if not stock_symbol:
            st.warning("Please enter a stock symbol")
            return
        
        # Initialize agents if not already done
        if not st.session_state.agents_initialized:
            with st.spinner("Initializing AI agents..."):
                if not initialize_agents():
                    return
        
        # Prepare query based on analysis type
        if analysis_type == "Complete Analysis (Recommendations + News)":
            query = f"Summarize analyst recommendation and share latest news for {stock_symbol}"
        elif analysis_type == "Analyst Recommendations Only":
            query = f"Provide detailed analyst recommendations for {stock_symbol}"
        elif analysis_type == "Latest News Only":
            query = f"Share the latest news and market sentiment for {stock_symbol}"
        elif analysis_type == "Stock Fundamentals":
            query = f"Analyze the fundamental metrics and financial health of {stock_symbol}"
        elif analysis_type == "Custom Query":
            query = custom_query if custom_query else f"Analyze {stock_symbol}"
        
        # Execute analysis
        with st.spinner(f"Analyzing {stock_symbol}... This may take a moment"):
            try:
                # Create a container for the response
                response_container = st.container()
                
                # Use the multi-agent system
                response = st.session_state.multi_ai_agents.run(query)
                
                # Display the response
                with response_container:
                    st.success(f"Analysis completed for {stock_symbol}")
                    
                    # Clean and display the response
                    cleaned_response = clean_response(str(response.content))
                    st.markdown(cleaned_response)
                    
                    # Add a download button for the response
                    st.download_button(
                        label="📥 Download Analysis",
                        data=cleaned_response,
                        file_name=f"{stock_symbol}_analysis.md",
                        mime="text/markdown"
                    )
            
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Please check your API keys and internet connection.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <p><strong>Multi-Agent Stock Analysis Platform</strong></p>
        <p>Powered by Phi Framework, Groq, and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()