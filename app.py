import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib # 日本語表示用

# --- ページ設定 ---
st.set_page_config(page_title="散布図・相関関係可視化ツール", layout="wide")

# タイトル
st.title("📊 散布図・相関関係可視化ツール")
st.markdown("""
このツールでは、2つのデータの関係性（相関）を散布図と相関係数で確認できます。
左のメニューから分析したいテーマを選んでみましょう！
""")

# --- サイドバー：設定パネル ---
st.sidebar.header("🕹️ 操作パネル")

# データセットの選択
dataset_type = st.sidebar.radio(
    "分析するデータを選んでください",
    (
        "📱 スマホ時間とテスト点数 (負の相関)", 
        "👟 身長と靴のサイズ (正の相関)", 
        "🎲 出席番号とテスト点数 (無相関)"
    )
)

st.sidebar.divider()
num_students = st.sidebar.slider("生徒数（データ件数）", min_value=10, max_value=200, value=50, step=10)
generate_btn = st.sidebar.button("🔄 データを再生成")

# --- データ生成関数 ---
@st.cache_data # 同じ設定なら再計算しない
def generate_correlation_data(n, dtype):
    np.random.seed(42) # 初回表示を固定（ボタンを押すと変わるように後で調整）
    
    # ボタンが押されたらシードをランダムにして新しいデータを作る
    if 'regen_trigger' not in st.session_state:
        st.session_state['regen_trigger'] = 0
    np.random.seed(42 + st.session_state['regen_trigger'])

    df = pd.DataFrame({'生徒ID': range(1, n + 1)})

    if dtype == "📱 スマホ時間とテスト点数 (負の相関)":
        usage_time = np.round(np.random.uniform(0.5, 6.0, n), 1)
        scores = 100 - (usage_time * 12) + np.random.normal(0, 8, n)
        df['スマホ利用時間(時間)'] = usage_time
        df['テストの点数'] = np.clip(scores, 0, 100).astype(int)

    elif dtype == "👟 身長と靴のサイズ (正の相関)":
        heights = np.round(np.random.normal(165, 8, n), 1)
        # 身長から靴のサイズを計算（少しばらつきを入れる）
        shoes = heights * 0.15 + np.random.normal(0, 1.0, n)
        # 靴のサイズを0.5刻みにする
        df['身長(cm)'] = heights
        df['靴のサイズ(cm)'] = np.round(shoes * 2) / 2

    elif dtype == "🎲 出席番号とテスト点数 (無相関)":
        # まったく関係のないランダムな点数
        scores = np.random.normal(60, 15, n)
        df['出席番号'] = range(1, n + 1)
        df['テストの点数'] = np.clip(scores, 0, 100).astype(int)

    return df

# ボタンが押されたらトリガーを更新してデータを再生成
if generate_btn:
    if 'regen_trigger' in st.session_state:
        st.session_state['regen_trigger'] += 1
    else:
        st.session_state['regen_trigger'] = 1

# データの取得
df = generate_correlation_data(num_students, dataset_type)


# --- データ分析・可視化エリア ---

st.header(f"テーマ: {dataset_type}")

# 1. データプレビューと基本統計量
col_data1, col_data2 = st.columns(2)
with col_data1:
    st.subheader("📋 データプレビュー（最初の5件）")
    st.dataframe(df.head(), use_container_width=True)

with col_data2:
    st.subheader("🔢 基本統計量")
    st.dataframe(df.describe(), use_container_width=True)

# 2. 散布図と相関係数
st.subheader("📈 散布図と相関関係の分析")

# 数値列の抽出（ID列はグラフから除外）
numeric_cols = [col for col in df.columns if col != '生徒ID']

if len(numeric_cols) >= 2:
    # 軸の自動設定
    x_axis = numeric_cols[0]
    y_axis = numeric_cols[1]

    # グラフ描画と相関係数表示
    col_plot, col_corr = st.columns([2, 1])
    
    with col_plot:
        # Seabornで散布図を描画
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax, s=100, alpha=0.7)
        
        # オプション：回帰直線を表示
        show_reg = st.checkbox("回帰直線（傾向線）を表示する", value=True)
        if show_reg:
            sns.regplot(data=df, x=x_axis, y=y_axis, ax=ax, scatter=False, color='red', line_kws={'linestyle':'--'})
        
        ax.set_title(f"{x_axis} と {y_axis} の散布図", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)

    with col_corr:
        st.write("### 相関係数 ($r$)")
        # 相関係数の算出
        correlation = df[x_axis].corr(df[y_axis])
        
        # メトリック表示
        st.metric(label=f"{x_axis} vs {y_axis}", value=f"{correlation:.3f}")

        # 相関の強さの判定
        st.write("#### 判定:")
        r_abs = abs(correlation)
        if r_abs >= 0.7:
            st.success("✅ **強い相関**があります。")
            if correlation > 0: st.write("（正の相関：Xが増えるとYも増える傾向）")
            else: st.write("（負の相関：Xが増えるとYは減る傾向）")
        elif r_abs >= 0.4:
            st.info("⚠️ **中程度の相関**があります。")
        elif r_abs >= 0.2:
            st.warning("🧐 **弱い相関**があります。")
        else:
            st.error("✖️ **ほとんど相関**がありません。")

        st.markdown("""
        ---
        **【情報Ⅰのポイント】**
        相関係数が高いからといって、必ずしも**因果関係**（一方が原因でもう一方が結果）があるとは限りません。
        背景にある他の要因（第三の変数）がないか考えてみましょう。
        """)
