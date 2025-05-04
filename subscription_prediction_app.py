import pandas as pd
import pdfplumber
import re
import yfinance as yf
import streamlit as st
import base64
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def extract_transactions_from_pdf(pdf_file):
    transactions = []
    date_pattern = r"(\d{1,2} \w{3}, \d{4})"
    amount_pattern = r"([-+]?\d{1,3}(?:,\d{3})*\.\d{2})"

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                lines = text.split("\n")
                for line in lines:
                    date_match = re.search(date_pattern, line)
                    amount_match = re.findall(amount_pattern, line)

                    if date_match and amount_match:
                        date = date_match.group(1)
                        amount = float(amount_match[-1].replace(",", ""))
                        description = line.replace(date, "").strip()
                        description = re.sub(amount_pattern, "", description).strip()

                        transactions.append([date, description, amount])

    df = pd.DataFrame(transactions, columns=["Date", "Description", "Amount"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(inplace=True)

    return df


def categorize_transactions(df):
    categories = {
        "Entertainment": ["netflix", "spotify", "youtube", "disney+"],
        "Shopping": ["amazon", "ebay"],
        "Utilities": ["electric", "water", "internet"],
    }
    df["Category"] = "Other"
    for category, keywords in categories.items():
        mask = df["Description"].str.lower().str.contains("|".join(keywords), case=False, na=False)
        df.loc[mask, "Category"] = category
    return df


def detect_subscriptions(df):
    subscription_keywords = ["netflix", "spotify", "amazon prime", "youtube premium", "apple music", "hulu", "disney+", "patreon"]
    df["is_subscription"] = df["Description"].str.contains('|'.join(subscription_keywords), case=False, na=False)
    return df[df["is_subscription"]]
def add_local_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style> 
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function before rendering Streamlit elements
add_local_background("BACKGROUND.jpg")  # Change this to your actual file name
# Streamlit UI
st.title("Subscription Spending Tracker")
# Sidebar navigation
st.sidebar.title("📊 App Navigation")
app_mode = st.sidebar.radio("Choose a module:", ["Subscription Tracker & Predictor", "Stocks Prediction"])

if app_mode == "Subscription Tracker & Predictor":
    
    st.markdown("""
    ## Welcome to the Subscription Tracker & Predictor!
    
    This app helps you:
    - 📂 Upload and analyze your monthly subscriptions
    - 💡 Understand where your money goes
    - 🤖 Predict whether you'll exceed your budget using machine learning
    
    Use the sidebar to switch between features. Let's get started!
    """)

    
    budget = st.number_input("💰 Set Monthly Budget (PKR)", min_value=0, value=5000)
    # User Input Option
    option = st.radio("📥 Select input method:", ("Upload PDF", "Upload CSV", "Enter Manually"))

    df = None  

    if option == "Upload PDF":
        pdf_file = st.file_uploader("📂 Upload your bank statement (PDF)", type=["pdf"])
        if pdf_file:
            df = extract_transactions_from_pdf(pdf_file)

    elif option == "Upload CSV":
        csv_file = st.file_uploader("📂 Upload your transactions (CSV)", type=["csv"])
        if csv_file:
            df = pd.read_csv(csv_file)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    elif option == "Enter Manually":
        st.write("📝 Enter your transactions below:")
        manual_data = st.text_area("Enter transactions in the format: Date, Description, Amount (one per line)")

        if manual_data:
            data = [line.strip().split(",") for line in manual_data.split("\n") if line]
            df = pd.DataFrame(data, columns=["Date", "Description", "Amount"])
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
            df.dropna(inplace=True)


    if df is not None and not df.empty:
        
        df = categorize_transactions(df)
        
        
        sub_df = detect_subscriptions(df)
        total_spent = sub_df["Amount"].sum() if not sub_df.empty else 0

        
        df["Month"] = df["Date"].dt.strftime("%b %Y")  # Ensure proper month format
        monthly_spending = df.groupby("Month")["Amount"].sum().reset_index()
        monthly_spending.columns = ["Month", "Total Spent"]
        
    
        st.write("### 📅 Monthly Spending Summary")
        st.dataframe(monthly_spending)
        
    
        budget_exceeded = monthly_spending[monthly_spending["Total Spent"] > budget]
        if not budget_exceeded.empty:
            st.warning("⚠️ Budget exceeded for the following months:")
            st.dataframe(budget_exceeded)
        
        
        st.write(f"### 💰 Total Subscription Spending: PKR {total_spent:.2f}")
        st.dataframe(sub_df)
        
    
        if not sub_df.empty:
            sub_df["Month"] = sub_df["Date"].dt.strftime("%b %Y")  # Ensure proper month format
            monthly_subscription_spending = sub_df.groupby("Month")["Amount"].sum()
            
            if not monthly_subscription_spending.empty:
                st.write("### 📅 Monthly Subscription Spending")
                st.bar_chart(monthly_subscription_spending)
        
        
        category_spending = df.groupby("Category")["Amount"].sum()
        if not category_spending.empty:
            st.write("### 📊 Spending by Category")
            st.bar_chart(category_spending)
        else:
            st.write("No categorized transactions available.")

        # === 🧠 Step 1: Feature Engineering for ML ===
        st.markdown("## 🧠 Machine Learning Feature Engineering")
        if st.button("🔄 Generate Features for ML"):
            with st.spinner("Processing features..."):
                # Prepare ML dataset
                ml_df = sub_df.copy()
                ml_df["Month"] = ml_df["Date"].dt.to_period("M").astype(str)
                
                feature_df = ml_df.groupby("Month").agg(
                    Num_Subscriptions=("Description", "count"),
                    Total_Spend=("Amount", "sum"),
                    Avg_Sub_Amount=("Amount", "mean")
                ).reset_index()

                feature_df["Exceeded_Budget"] = (feature_df["Total_Spend"].abs() > budget).astype(int)

                st.success("✅ Features generated successfully!")
                st.write("### 🔍 ML Feature Data")
                st.dataframe(feature_df)

                # Save in session state for ML pipeline
                st.session_state["feature_df"] = feature_df
        # === 🧪 Step 2: Train/Test Split ===
        if "feature_df" in st.session_state:
            feature_df = st.session_state["feature_df"]
            
            st.markdown("##  Train/Test Split")
            if st.button("✂️ Split Data"):
                from sklearn.model_selection import train_test_split
                import plotly.express as px

                X = feature_df[["Num_Subscriptions", "Total_Spend", "Avg_Sub_Amount"]]
                y = feature_df["Exceeded_Budget"]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                st.session_state["X_train"] = X_train
                st.session_state["X_test"] = X_test
                st.session_state["y_train"] = y_train
                st.session_state["y_test"] = y_test

                st.success("✅ Data split into training and testing sets!")

                # Show pie chart of split sizes
                sizes = [len(X_train), len(X_test)]
                labels = ["Training Set", "Testing Set"]
                fig = px.pie(
                    values=sizes,
                    names=labels,
                    title="📊 Train/Test Split Ratio",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig)

        # === 🔮 Step 3: Forecast Next Month's Subscription Spending ===
        if "feature_df" in st.session_state:
            st.markdown("##  Forecast Next Month's Subscription Spending")

            if st.button("📈 Train Forecast Model"):
                from sklearn.linear_model import LinearRegression
                import numpy as np

                feature_df = st.session_state["feature_df"].copy()

                # Convert Month to numeric sequence
                feature_df["Month_Num"] = range(1, len(feature_df) + 1)

                # Features and target
                X = feature_df[["Month_Num", "Num_Subscriptions", "Avg_Sub_Amount"]]
                y = feature_df["Total_Spend"]

                # Train model
                model = LinearRegression()
                model.fit(X, y)

                # Predict next month
                next_month = pd.DataFrame({
                    "Month_Num": [len(feature_df) + 1],
                    "Num_Subscriptions": [X["Num_Subscriptions"].mean()],
                    "Avg_Sub_Amount": [X["Avg_Sub_Amount"].mean()]
                })

                predicted_spend = model.predict(next_month)[0]

                st.success(f"📊 Predicted Subscription Spending for Next Month: **PKR {predicted_spend:.2f}**")

                # Optional: Plot actual vs. predicted
                feature_df["Predicted"] = model.predict(X)
                mae = mean_absolute_error(y, feature_df["Predicted"])
                mse = mean_squared_error(y, feature_df["Predicted"])
                rmse = mse ** 0.5
                r2 = r2_score(y, feature_df["Predicted"])

                st.markdown("### 📊 Model Evaluation Metrics")
                st.write(f"**MAE:** {mae:.2f}")
                st.write(f"**MSE:** {mse:.2f}")
                st.write(f"**RMSE:** {rmse:.2f}")
                st.write(f"**R² Score:** {r2:.2f}")
                fig = px.line(feature_df, x="Month_Num", y=["Total_Spend", "Predicted"],
                                labels={"value": "Spending (PKR)", "Month_Num": "Month"},
                                title="🔁 Actual vs. Predicted Spending")
                st.plotly_chart(fig)


                    # Optional: Save model if you want to use it later
                st.session_state["model"] = model
                st.success("✅ Model saved in session state!")
    
    
elif app_mode == "Stocks Prediction":
    

    st.markdown("## 📈 Stocks Price Forecasting")

    ticker = st.text_input("🔍 Enter a stock ticker symbol (e.g., AAPL, TSLA, MSFT)", value="AAPL").strip().upper()

    # Ensure it's a single ticker (no commas or spaces)
    if "," in ticker or " " in ticker:
        st.error("❌ Please enter only one stock ticker symbol without spaces or commas.")
    else:
        if st.button("📥 Fetch Data"):
            try:
                stock_data = yf.download(ticker, start="2022-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'))
                stock_data.reset_index(inplace=True)
                
                if not stock_data.empty:
                    st.success(f"✅ Data fetched for {ticker}")
                    st.dataframe(stock_data.head())
                    print(stock_data.columns) 
                    date_column = 'Date'
                    close_column = 'Close'
                    
                    # Adjust for multi-level column names
                    # Assuming the format of the columns is "Close_{ticker}"

                    # Plot
                    st.write("### 📊 Closing Price Over Time")
                    if isinstance(stock_data.columns[0], tuple):
                        date_column = ('Date', '')
                    if isinstance(stock_data.columns[1], tuple):
                        close_column = ('Close', ticker)
                    fig = px.line(stock_data, x=date_column, y=close_column, title=f'{ticker} Closing Price')
                    st.plotly_chart(fig)

                    st.session_state["stock_data"] = stock_data
                    st.session_state["ticker"] = ticker
                else:
                    st.warning("⚠️ No data found for this ticker.")
            except Exception as e:
                st.error(f"❌ Error fetching data: {e}")

    
    st.markdown("## 🤖 Predict Future Prices")

    future_days = st.slider("🔮 Days to Predict", min_value=1, max_value=30, value=7)

    if "stock_data" in st.session_state and st.button("🚀 Train Prediction Model"):
        from sklearn.linear_model import LinearRegression
        import numpy as np

        stock_data = st.session_state["stock_data"].copy()
        stock_data["Date"] = pd.to_datetime(stock_data["Date"])
        stock_data = stock_data[["Date", "Close"]].dropna()

        # Convert dates to numerical values for regression
        stock_data["Days"] = (stock_data["Date"] - stock_data["Date"].min()).dt.days
        X = stock_data[["Days"]]
        y = stock_data["Close"]
        
        split_idx = int(len(stock_data) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = LinearRegression()
        model.fit(X, y)
        
        y_pred_test = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_test)
        
        st.markdown("### 📊 Model Evaluation (on Test Data)")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**MSE:** {mse:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        # Predict future days
        last_day = stock_data["Days"].max()
        future_X = pd.DataFrame({"Days": range(last_day + 1, last_day + future_days + 1)})
        future_pred = model.predict(future_X).flatten()  # Ensure 1D


        # Convert prediction days back to actual dates
        future_dates = pd.date_range(start=stock_data["Date"].max() + pd.Timedelta(days=1), periods=future_days)
        forecast_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_pred})
        st.success("✅ Model trained and future prices predicted!")
        st.write("### 🔮 Future Stock Price Forecast")
        st.dataframe(forecast_df)

        # Plot future forecast
        full_plot = pd.concat([
            stock_data[["Date", "Close"]].rename(columns={"Close": "Price"}),
            forecast_df.rename(columns={"Predicted Close": "Price"})
        ])
        full_plot["Type"] = ["Historical"] * len(stock_data) + ["Forecast"] * len(forecast_df)

        fig = px.line(full_plot, x="Date", y="Price", color="Type", title="📈 Stock Price Forecast")
        st.plotly_chart(fig) 
