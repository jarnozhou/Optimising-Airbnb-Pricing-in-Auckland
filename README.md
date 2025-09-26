# Optimising Airbnb Pricing in Auckland

## 📌 Overview
This project explores how to **optimise Airbnb pricing in Auckland** by identifying key property features that influence price and uncovering **seasonal pricing trends**.  
The key problems addressed are:
1. Which property attributes (location, amenities, reviews, host status, etc.) most significantly impact Airbnb prices in Auckland?  
2. How do listing prices fluctuate across months, and what seasonal patterns exist?  

By answering these questions, the project provides **data-driven insights and strategies** to help Airbnb hosts optimise pricing, stay competitive, and maximise revenue.

---

## 📊 Dataset Description
- **Source:** 9,901 Airbnb listings in Auckland with ~80 variables, including property details, host attributes, amenities, reviews, and pricing.  
- **Transformation & Cleaning:**  
  - Filtered for Auckland listings only.  
  - Merged 12 months of historical price data.  
  - Removed non-essential columns, converted inconsistent formats, and encoded variables.  
  - Imputed missing values (e.g., bathrooms, review scores, prices) using median-based rules.  
  - Consolidated 100+ amenities into 7 structured categories (Basic, Hygiene, Safety, Kitchen & Dining, Entertainment, Outdoor & View, Extras).  
  - Outliers handled by capping price at the 95th percentile and excluding high volatility (>0.6).  
  - Created new features: median price, price range, and price volatility.  

- **Final Dataset:**  
  - **6,124 listings**  
  - **49 cleaned variables**  
  - No missing values, consistent formats, and realistic price ranges  

---

## 🔬 Methods
1. **Descriptive Statistics & EDA**  
   - Distribution analysis of room type, accommodates, amenities, reviews, and regional patterns.  
   - Correlation analysis to remove redundant variables.  

2. **Linear Regression Analysis**  
   - Modelled the relationship between price and predictors (room type, accommodates, host verification, reviews, amenities, etc.).  
   - Checked assumptions (multicollinearity, VIF, collinearity).  

3. **Time Series Analysis (Feb 2024 – Jan 2025)**  
   - Examined monthly average price trends.  
   - Compared seasonal variation across **regions** and **room types**.  
   - Performed **ANOVA tests** to test significance of monthly price variation.  

---

## 📈 Results
- **Price Drivers (Regression):**  
  - **Room Type**: Strongest factor (private/shared rooms priced 40–70% lower than entire homes).  
  - **Accommodates**: Each additional guest capacity increases price ~37%.  
  - **Reviews**: Higher location & overall ratings increase price, while “value” ratings correlate with lower prices.  
  - **Amenities**: Outdoor views and extras (pool, BBQ, etc.) increase price, while hygiene/kitchen amenities had weaker effects.  
  - **Host Attributes**: Superhost status had little effect, but verified hosts priced slightly lower.  

- **Seasonal Trends (Time Series):**  
  - Prices peak in **January** (summer/holiday season) and **May** (autumn travel demand).  
  - Prices dip between **September–December** (lower demand).  
  - **Regions:** Tourist areas (Rodney, Waitematā & Gulf) show higher volatility; suburban areas (Manukau) are more stable.  
  - **Room Types:** Entire homes and hotel rooms fluctuate more with seasonality; private/shared rooms remain stable.  

- **ANOVA Tests:** Confirmed significant monthly price variation (p < 2e-16), but patterns are consistent across regions and room types.  

---

## 💡 Recommendations
1. **Dynamic Pricing**  
   - Adjust prices in real-time for seasonality, events, and demand.  
   - Use flexible pricing for weekends, last-minute stays, and early bookings.  
   - Apply automated tools (e.g., Airbnb Smart Pricing, PriceLabs).  

2. **Strategies for Established Hosts**  
   - Invest in high-impact amenities (outdoor views, entertainment, extras).  
   - Enhance guest experience (e.g., local guidebooks, complimentary services).  
   - Optimise listings with updated descriptions, photos, and targeted promotions.  

3. **Strategies for New Hosts**  
   - Start with competitive launch pricing and introductory discounts.  
   - Highlight cleanliness, safety, and location advantages in listings.  
   - Build credibility with strong guest reviews before raising prices.  

---

## 🙋‍♂️ Contribution
I was responsible for both the **technical modelling** and **strategic recommendations**:  
- **Linear Regression Analysis** – to identify and quantify the impact of key variables on pricing, providing interpretable insights for decision-making.  
- **Time Series Analysis** – to uncover seasonal patterns and statistically validate price fluctuations across months, regions, and room types.  
I chose these methods because they complement each other: **regression explains what drives price differences across listings, while time series highlights when and how prices fluctuate throughout the year**. Together, they provided the foundation for evidence-based recommendations.
- **Recommendations** – brainstormed actionable strategies for both new and established hosts, directly linking the modelling results to practical pricing approaches.  

