import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
import json
import re

st.set_page_config(
    page_title="Diabetes Risk Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
    }
    .insight-box {
        background-color: #f9f9f9;
        padding: 12px;
        border-radius: 6px;
        margin: 8px 0;
        border-left: 4px solid #1f77b4;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    .actionable-box {
        border-left-color: #d32f2f;
    }
    .confounded-box {
        border-left-color: #999;
        background-color: #fafafa;
    }
    .metric-label {
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 8px;
        margin-bottom: 2px;
    }
    .benchmark-text {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0px;
        margin-top: -2px;
        line-height: 1.2;
    }
    .metric-spacing {
        margin-bottom: 14px;
    }
    .section-header {
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 16px;
        margin-bottom: 10px;
        color: #1f77b4;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 6px;
    }
    .metric-row {
        margin-bottom: 10px;
    }
    .metric-value {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 1px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = pd.read_csv('files/webapp_data.csv')
    fips = pd.read_excel('files/US_FIPS_Codes.xls')
    return data, fips

@st.cache_data
def load_geojson():
    with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
        counties = json.load(response)
    return counties

FEATURE_DESCRIPTIONS = {
    'pct_without_insurance': {
        'name': 'Uninsured Population (%)',
        'modifiable': True,
        'ideal': 'lower'
    },
    'pct_with_public_insurance': {
        'name': 'Public Insurance Coverage (%)',
        'modifiable': True,
        'ideal': 'context-dependent'
    },
    'pct_obesity': {
        'name': 'Obesity (%)',
        'modifiable': True,
        'ideal': 'lower'
    },
    'pct_diabetes': {
        'name': 'Diabetes Prevalence (%)',
        'modifiable': False,
        'ideal': 'lower'
    },
    'pct_low_physical_activity': {
        'name': 'Low Physical Activity (%)',
        'modifiable': True,
        'ideal': 'lower'
    },
    'pct_current_smoking': {
        'name': 'Current Smoking (%)',
        'modifiable': True,
        'ideal': 'lower'
    },
    'pct_high_blood_pressure': {
        'name': 'High Blood Pressure (%)',
        'modifiable': True,
        'ideal': 'lower'
    },
    'median_household_income': {
        'name': 'Median Household Income ($)',
        'modifiable': False,
        'ideal': 'higher'
    },
    'per_capita_income': {
        'name': 'Per Capita Income ($)',
        'modifiable': False,
        'ideal': 'higher'
    },
    'unemployment_rate': {
        'name': 'Unemployment Rate (%)',
        'modifiable': True,
        'ideal': 'lower'
    },
    'primary_care_physicians_per_1k': {
        'name': 'Primary Care Physicians (per 1k)',
        'modifiable': False,
        'ideal': 'higher'
    },
    'nurse_practitioners_per_1k': {
        'name': 'Nurse Practitioners (per 1k)',
        'modifiable': False,
        'ideal': 'higher'
    },
    'critical_access_hospitals_per_1k': {
        'name': 'Critical Access Hospitals (per 1k)',
        'modifiable': False,
        'ideal': 'higher'
    },
    'hospitals_diabetes_prevention_per_1k': {
        'name': 'Diabetes Prevention Programs (per 1k)',
        'modifiable': False,
        'ideal': 'higher'
    },
    'hospitals_nutrition_programs_per_1k': {
        'name': 'Nutrition Programs (per 1k)',
        'modifiable': False,
        'ideal': 'higher'
    },
    'pct_annual_checkup': {
        'name': 'Annual Checkup Rate (%)',
        'modifiable': True,
        'ideal': 'higher'
    },
    'pct_no_health_insurance_places': {
        'name': 'No Insurance Options Nearby (%)',
        'modifiable': True,
        'ideal': 'lower'
    }
}

def get_risk_category(risk_score):
    if risk_score < 1.0:
        return "Low Risk", "risk-low", "#388e3c"
    elif risk_score < 1.5:
        return "Medium Risk", "risk-medium", "#f57c00"
    else:
        return "High Risk", "risk-high", "#d32f2f"

def calculate_benchmarks(data):
    benchmarks = {}
    
    for col in FEATURE_DESCRIPTIONS.keys():
        if col in data.columns:
            benchmarks[col] = {
                'national_median': data[col].median(),
                'national_p25': data[col].quantile(0.25),
                'national_p75': data[col].quantile(0.75),
                'state_medians': data.groupby('State')[col].median().to_dict()
            }
    
    return benchmarks

def format_benchmark_comparison(county_value, col, benchmarks, state=None):
    if col not in benchmarks:
        return ""
    
    bench = benchmarks[col]
    nat_median = bench['national_median']
    
    if state and state in bench['state_medians']:
        state_median = bench['state_medians'][state]
        return f"State: {state_median:.0f} | National: {nat_median:.0f}"
    else:
        return f"National: {nat_median:.0f}"

def create_county_map(data_with_fips, selected_fips=None, high_risk_only=False):
    counties_geojson = load_geojson()
    
    map_data = data_with_fips.copy()
    
    if high_risk_only:
        map_data = map_data[map_data['predicted_risk'] > 1.5]
    
    map_data['fips'] = map_data['fips'].astype(str).str.zfill(5)
    
    fig = px.choropleth(
        map_data,
        geojson=counties_geojson,
        locations='fips',
        color='predicted_risk',
        color_continuous_scale='RdYlGn_r',
        range_color=[data_with_fips['predicted_risk'].min(), data_with_fips['predicted_risk'].max()],
        scope="usa",
        labels={'predicted_risk': 'Risk Score'},
        hover_data={
            'County': True,
            'State': True,
            'predicted_risk': ':.2f',
            'fips': False
        },
        hover_name='County'
    )
    
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=600,
        coloraxis_colorbar=dict(
            title="Risk Score<br>(per 1,000)",
            thicknessmode="pixels",
            thickness=15,
            lenmode="pixels",
            len=300,
        )
    )
    
    if selected_fips:
        selected_county = data_with_fips[data_with_fips['fips'] == selected_fips]
        if not selected_county.empty:
            fig.add_trace(go.Choropleth(
                geojson=counties_geojson,
                locations=[selected_fips],
                z=[1],
                colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
                showscale=False,
                marker_line_color='yellow',
                marker_line_width=3,
                hoverinfo='skip'
            ))
    
    return fig

def create_risk_drivers_chart(county_data, shap_cols, top_n=10):
    shap_data = []
    for col in shap_cols:
        if col.startswith('shap_'):
            feature = col.replace('shap_', '')
            shap_value = county_data[col].values[0]
            feature_value = county_data[feature].values[0]
            
            if feature in FEATURE_DESCRIPTIONS:
                feature_info = FEATURE_DESCRIPTIONS[feature]
                shap_data.append({
                    'feature': feature_info['name'],
                    'shap_value': shap_value,
                    'feature_value': feature_value,
                    'modifiable': feature_info['modifiable'],
                    'raw_feature': feature
                })
    
    shap_df = pd.DataFrame(shap_data)
    shap_df['abs_shap'] = shap_df['shap_value'].abs()
    shap_df = shap_df.sort_values('abs_shap', ascending=False).head(top_n)
    
    colors = ['#d32f2f' if x < 0 else '#388e3c' for x in shap_df['shap_value']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=shap_df['feature'],
        x=shap_df['shap_value'],
        orientation='h',
        marker=dict(color=colors),
        text=shap_df['shap_value'].round(3),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Strength of association: %{x:.3f}<br>Value: %{customdata}<extra></extra>',
        customdata=shap_df['feature_value'].round(2)
    ))
    
    fig.update_layout(
        title='SHAP Breakdown for Risk Factors',
        xaxis_title='SHAP Value (association strength)',
        yaxis_title='',
        height=450,
        margin=dict(l=200, r=50, t=60, b=60),
        showlegend=False
    )
    
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
    
    return fig, shap_df

def display_county_detail(county_data, shap_cols, benchmarks):
    county_name = county_data['County'].values[0]
    state_name = county_data['State'].values[0]
    risk_score = county_data['predicted_risk'].values[0]
    rank = county_data['risk_rank'].values[0]
    percentile = county_data['risk_percentile'].values[0]
    
    category, css_class, color = get_risk_category(risk_score)
    
    st.markdown(f"## {county_name}, {state_name}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Predicted Risk",
            help='Deaths with diabetes as underlying cause, per 1,000 population',
            value=f"{risk_score:.2f}",
        )
    
    with col2:
        st.metric(
            label="Risk Category",
            value=category,
        )    

    with col3:
        st.metric(
            label="National Rank",
            value=f"#{rank}",
            delta=f"Top {percentile:.1f}%",
            delta_color="off"
        )
    
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    
    # LEFT COLUMN: Risk drivers + actionable factors
    with col_left:
        st.markdown("### Risk Factors by Strength of Association")
        fig, shap_df = create_risk_drivers_chart(county_data, shap_cols, top_n=12)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Focus Areas for Intervention")
        st.markdown("""
        These factors show the strongest associations with diabetes mortality in this county. 
        They represent areas where public health interventions could influence outcomes.
        """)
        
        actionable_factors = shap_df[shap_df['modifiable'] == True].head(5)
        
        for _, row in actionable_factors.iterrows():
            bench_text = format_benchmark_comparison(row['feature_value'], row['raw_feature'], benchmarks, state_name)
            
            # Determine if it's above/below benchmark
            feature_col = row['raw_feature']
            if feature_col in benchmarks:
                nat_med = benchmarks[feature_col]['national_median']
                comparison_text = f"Above national avg" if row['feature_value'] > nat_med else f"Below national avg"
                comparison_color = "#d32f2f" if row['feature_value'] > nat_med else "#388e3c"
            else:
                comparison_text = ""
                comparison_color = "#666"
            
            st.markdown(
                f"<div class='insight-box actionable-box'>"
                f"<b>{row['feature']}</b><br>"
                f"<small>Current: {row['feature_value']:.0f} | {bench_text}</small><br>"
                f"<small style='color:{comparison_color};'>→ {comparison_text}</small>"
                f"</div>",
                unsafe_allow_html=True
            )
    
    # RIGHT COLUMN: County profile vs benchmarks
    with col_right:
        st.markdown("### County Profile vs. Benchmarks")
        
        obesity = county_data['pct_obesity'].values[0]
        physical_activity = county_data['pct_low_physical_activity'].values[0]
        diabetes = county_data['pct_diabetes'].values[0]
        smoking = county_data['pct_current_smoking'].values[0]
        uninsured = county_data['pct_without_insurance'].values[0]
        checkups = county_data['pct_annual_checkup'].values[0]
        physicians = county_data['primary_care_physicians_per_1k'].values[0]
        nurses = county_data['nurse_practitioners_per_1k'].values[0]
        income = county_data['median_household_income'].values[0]
        unemployment = county_data['unemployment_rate'].values[0]
        
        st.markdown("<div class='section-header'>Health Behaviors</div>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='metric-row'><div class='metric-label'>Obesity</div><div class='metric-value'>{obesity:.0f}%</div><div class='benchmark-text'>{format_benchmark_comparison(obesity, 'pct_obesity', benchmarks, state_name)}</div></div>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='metric-row'><div class='metric-label'>Current Smoking</div><div class='metric-value'>{smoking:.0f}%</div><div class='benchmark-text'>{format_benchmark_comparison(smoking, 'pct_current_smoking', benchmarks, state_name)}</div></div>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='metric-row'><div class='metric-label'>Physical Inactivity</div><div class='metric-value'>{physical_activity:.0f}%</div><div class='benchmark-text'>{format_benchmark_comparison(physical_activity, 'pct_low_physical_activity', benchmarks, state_name)}</div></div>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='metric-row'><div class='metric-label'>Diabetes Prevalence</div><div class='metric-value'>{diabetes:.0f}%</div><div class='benchmark-text'>{format_benchmark_comparison(diabetes, 'pct_diabetes', benchmarks, state_name)}</div></div>", unsafe_allow_html=True)
        
        st.markdown("<div class='section-header'>Healthcare Access</div>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='metric-row'><div class='metric-label'>Uninsured</div><div class='metric-value'>{uninsured:.0f}%</div><div class='benchmark-text'>{format_benchmark_comparison(uninsured, 'pct_without_insurance', benchmarks, state_name)}</div></div>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='metric-row'><div class='metric-label'>Annual Checkups</div><div class='metric-value'>{checkups:.0f}%</div><div class='benchmark-text'>{format_benchmark_comparison(checkups, 'pct_annual_checkup', benchmarks, state_name)}</div></div>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='metric-row'><div class='metric-label'>Primary Care Physicians</div><div class='metric-value'>{physicians:.0f} per 1k</div><div class='benchmark-text'>{format_benchmark_comparison(physicians, 'primary_care_physicians_per_1k', benchmarks, state_name)}</div></div>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='metric-row'><div class='metric-label'>Nurse Practitioners</div><div class='metric-value'>{nurses:.0f} per 1k</div><div class='benchmark-text'>{format_benchmark_comparison(nurses, 'nurse_practitioners_per_1k', benchmarks, state_name)}</div></div>", unsafe_allow_html=True)
        
        st.markdown("<div class='section-header'>Socioeconomic</div>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='metric-row'><div class='metric-label'>Median Income</div><div class='metric-value'>${income:,.0f}</div><div class='benchmark-text'>{format_benchmark_comparison(income, 'median_household_income', benchmarks, state_name)}</div></div>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='metric-row'><div class='metric-label'>Unemployment</div><div class='metric-value'>{unemployment:.0f}%</div><div class='benchmark-text'>{format_benchmark_comparison(unemployment, 'unemployment_rate', benchmarks, state_name)}</div></div>", unsafe_allow_html=True)

data, fips_data = load_data()

benchmarks = calculate_benchmarks(data)
    
def minimal_fix(name):
    n = name.strip().upper()
    
    n = re.sub(r'^ST\.\s*', 'ST ', n)
    n = re.sub(r'^STE\.\s*', 'STE ', n)
    
    fix_map = {
        "LASALLE": "LA SALLE",
        "DESOTO": "DE SOTO",
        "DEWITT": "DE WITT",
        "CHARLES": "CHARLES CITY",
        "JAMES": "JAMES CITY",
        "PRINCE GEORGE'S": "PRINCE GEORGES",
        "QUEEN ANNE'S": "QUEEN ANNES",
        "ST MARY'S": "ST MARYS",
        "DUPAGE": "DU PAGE",
        "DEKALB": "DE KALB"
    }
    
    if n in fix_map:
        n = fix_map[n]
    
    return n

data['County_clean'] = data['County'].apply(minimal_fix)
data['State_clean'] = data['State'].str.strip().str.upper()
data.loc[(data['County_clean'] == "DE KALB") & (data['State_clean'] == "TENNESSEE"), 'County_clean'] = "DEKALB"
data.loc[(data['County_clean'] == "DE KALB") & (data['State_clean'] == "MISSOURI"), 'County_clean'] = "DEKALB"
data.loc[(data['County_clean'] == "CHARLES CITY") & (data['State_clean'] == "MARYLAND"), 'County_clean'] = "CHARLES"

fips_data['County_clean'] = fips_data['County Name'].str.strip().str.upper()
fips_data['State_clean'] = fips_data['State'].str.strip().str.upper()

fips_data['fips'] = (fips_data['FIPS State'].astype(str).str.zfill(2) + 
                        fips_data['FIPS County'].astype(str).str.zfill(3))
    
data_with_fips = data.merge(
    fips_data[['County_clean', 'State_clean', 'fips']],
    on=['County_clean', 'State_clean'],
    how='left'
)

data_with_fips = data_with_fips.dropna(subset=['fips'])

shap_cols = [col for col in data.columns if col.startswith('shap_')]

if 'selected_fips' not in st.session_state:
    st.session_state.selected_fips = None

with st.sidebar:
    st.title("🔍 County Search")
    
    states = sorted(data_with_fips['State'].unique())
    selected_state = st.selectbox("Select State", ['All States'] + states, key='state_filter')
    
    if selected_state != 'All States':
        filtered_data = data_with_fips[data_with_fips['State'] == selected_state]
    else:
        filtered_data = data_with_fips
    
    counties = sorted(filtered_data['County'].unique())
    selected_county_name = st.selectbox("Select County", counties, key='county_filter')
    
    county_row = filtered_data[filtered_data['County'] == selected_county_name].iloc[0]
    st.session_state.selected_fips = county_row['fips']
    
    st.markdown("---")
    
    st.markdown("### Map Filters")
    show_high_risk_only = st.checkbox("Show high-risk counties only (>1.5)", value=False)
    
    if show_high_risk_only:
        map_data = data_with_fips[data_with_fips['predicted_risk'] > 1.5]
    else:
        map_data = data_with_fips
    
    st.markdown(f"**Showing:** {len(map_data):,} counties")

st.markdown("### Interactive Risk Map")

fig = create_county_map(map_data, st.session_state.selected_fips, show_high_risk_only)

selected_points = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="county_map")

if selected_points and selected_points.selection and selected_points.selection.points:
    point = selected_points.selection.points[0]
    if 'location' in point:
        st.session_state.selected_fips = point['location']

if st.session_state.selected_fips:
    selected_county_data = data_with_fips[data_with_fips['fips'] == st.session_state.selected_fips]
    
    if not selected_county_data.empty:
        st.markdown("---")
        display_county_detail(selected_county_data, shap_cols, benchmarks)
    else:
        st.warning("Selected county data not found.")
else:
    st.info("Click on a county on the map above to view detailed analysis.")

