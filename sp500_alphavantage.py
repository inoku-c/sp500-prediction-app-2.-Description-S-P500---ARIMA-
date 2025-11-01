import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(page_title="S&P500翌日予測", layout="wide")

# タイトルとヘッダー
st.title("📈 S&P500 翌日価格予測アプリ")
st.markdown("**ARIMAモデル + Alpha Vantage API**")
st.markdown("---")

# Alpha Vantage APIキーの取得
st.sidebar.header("⚙️ API設定")

# API Key入力
api_key = st.sidebar.text_input(
    "Alpha Vantage API Key", 
    type="password",
    help="無料取得: https://www.alphavantage.co/support/#api-key"
)

if not api_key:
    st.info("👈 Alpha Vantage API Keyを入力してください")
    st.markdown("""
    ## 🔑 API Keyの取得方法
    
    1. [Alpha Vantage](https://www.alphavantage.co/support/#api-key) にアクセス
    2. メールアドレスを入力して無料API Keyを取得
    3. 左のサイドバーにAPI Keyを入力
    
    **無料プラン**: 1日500リクエスト、1分5リクエストまで
    
    ## 📊 このアプリについて
    
    - **データソース**: Alpha Vantage API（信頼性の高い金融データAPI）
    - **対象**: SPY（S&P500 ETF）- S&P500指数を正確に追跡
    - **予測手法**: ARIMA自動最適化モデル
    - **予測期間**: 翌日の終値
    
    ### ✨ 特徴
    
    - 安定したデータ取得
    - API制限を考慮した設計
    - 自動パラメータ最適化
    - 95%信頼区間付き予測
    """)
    st.stop()

# サイドバー設定
st.sidebar.markdown("---")
st.sidebar.subheader("📊 データ設定")

# データ取得期間
period_options = {
    "1ヶ月": "compact",  # 100日分
    "完全データ": "full"  # 20年分
}
selected_period = st.sidebar.selectbox("データ取得量", list(period_options.keys()), index=0)
outputsize = period_options[selected_period]

# 最適化設定
st.sidebar.subheader("🤖 モデル設定")
st.sidebar.info("パラメータは自動で最適化されます")

with st.sidebar.expander("詳細設定"):
    max_p = st.slider("最大p値", 1, 5, 3)
    max_q = st.slider("最大q値", 1, 5, 3)
    max_d = st.slider("最大d値", 0, 2, 1)

# データ取得ボタン
st.sidebar.markdown("---")
run_analysis = st.sidebar.button("🚀 翌日価格を予測", type="primary", use_container_width=True)

# Alpha Vantageからデータ取得
@st.cache_data(ttl=3600)  # 1時間キャッシュ
def get_sp500_data(api_key, outputsize='compact'):
    """Alpha Vantage APIからSPYデータを取得"""
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=SPY&outputsize={outputsize}&apikey={api_key}'
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # エラーチェック
        if "Error Message" in data:
            return None, "無効なAPIリクエストです"
        
        if "Note" in data:
            return None, "API制限に達しました。1分後に再試行してください"
        
        if "Time Series (Daily)" not in data:
            return None, f"データの取得に失敗しました: {data.get('Information', '不明なエラー')}"
        
        # データフレームに変換
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # カラム名を変更
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # 数値型に変換
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        return df, None
    
    except requests.exceptions.Timeout:
        return None, "タイムアウト: APIサーバーからの応答がありません"
    except requests.exceptions.RequestException as e:
        return None, f"接続エラー: {str(e)}"
    except Exception as e:
        return None, f"予期しないエラー: {str(e)}"

# 自動パラメータ最適化
def optimize_arima(data, max_p=3, max_q=3, max_d=1):
    """AICを基準に最適なARIMAパラメータを探索"""
    best_aic = np.inf
    best_order = None
    best_model = None
    
    for p in range(0, max_p + 1):
        for d in range(0, max_d + 1):
            for q in range(0, max_q + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        best_model = fitted
                except:
                    continue
    
    return best_model, best_order, best_aic

# メインコンテンツ
if run_analysis:
    with st.spinner("Alpha Vantage APIからデータを取得中..."):
        # データ取得
        sp500_data, error = get_sp500_data(api_key, outputsize)
        
        if error:
            st.error(f"❌ {error}")
            if "API制限" in error:
                st.info("💡 無料プランは1分に5リクエストまでです。少し待ってから再試行してください。")
            st.stop()
        
        if sp500_data is None or sp500_data.empty:
            st.error("データの取得に失敗しました")
            st.stop()
        
        # 終値データ
        data = sp500_data['Close'].dropna()
        
        if len(data) < 30:
            st.error(f"データが不足しています（取得: {len(data)}件）")
            st.stop()
        
        # 最新情報
        latest_date = data.index[-1]
        latest_price = float(data.iloc[-1])
        prev_price = float(data.iloc[-2])
        recent_change = ((latest_price - prev_price) / prev_price) * 100
        next_day = latest_date + timedelta(days=1)
        
        # データ情報表示
        st.success(f"✅ データ取得成功: {len(data)}日分のデータ")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("データ数", f"{len(data)}日")
        col2.metric("最新価格 (SPY)", f"${latest_price:.2f}")
        col3.metric("前日比", f"{recent_change:+.2f}%")
        col4.metric("予測日", next_day.strftime('%Y-%m-%d'))
        
        st.markdown("---")
        
        # データプロット
        st.subheader("📊 SPY (S&P500 ETF) 終値データ")
        recent_data = data.tail(min(60, len(data)))
        
        fig_data = go.Figure()
        fig_data.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data.values,
            mode='lines+markers',
            name='SPY',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        fig_data.update_layout(
            xaxis_title="日付",
            yaxis_title="価格 (USD)",
            hovermode='x unified',
            height=350
        )
        st.plotly_chart(fig_data, use_container_width=True)
        
        # ARIMAモデル最適化
        st.markdown("---")
        st.subheader("🤖 ARIMAモデル自動最適化")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("パラメータ探索中...")
            progress_bar.progress(30)
            
            # 最適化
            best_model, best_order, best_aic = optimize_arima(
                data, max_p=max_p, max_q=max_q, max_d=max_d
            )
            
            if best_model is None:
                st.error("最適なモデルが見つかりませんでした")
                st.stop()
            
            progress_bar.progress(70)
            status_text.text("予測中...")
            
            p_opt, d_opt, q_opt = best_order
            
            # 予測
            forecast_result = best_model.get_forecast(steps=1)
            next_day_prediction = float(forecast_result.predicted_mean.iloc[0])
            conf_int = forecast_result.conf_int().iloc[0]
            conf_int_lower = float(conf_int.iloc[0])
            conf_int_upper = float(conf_int.iloc[1])
            
            progress_bar.progress(100)
            status_text.text("完了!")
            
            st.success(f"✅ 最適モデル: ARIMA({p_opt}, {d_opt}, {q_opt})")
            
            # 予測結果
            st.markdown("---")
            st.subheader("📈 予測結果")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                price_change = next_day_prediction - latest_price
                price_change_pct = (price_change / latest_price) * 100
                
                st.markdown("### 🎯 翌日予測価格 (SPY)")
                st.markdown(f"# ${next_day_prediction:.2f}")
                
                if price_change > 0:
                    st.markdown(f"<h3 style='color: green;'>▲ ${price_change:.2f} (+{price_change_pct:.2f}%)</h3>", 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='color: red;'>▼ ${abs(price_change):.2f} ({price_change_pct:.2f}%)</h3>", 
                              unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 詳細情報
            col1, col2, col3 = st.columns(3)
            col1.metric("現在価格", f"${latest_price:.2f}")
            col2.metric("予測価格", f"${next_day_prediction:.2f}")
            col3.metric("予測日", next_day.strftime('%Y/%m/%d'))
            
            # 信頼区間
            st.markdown("### 📊 95% 信頼区間")
            col1, col2, col3 = st.columns(3)
            col1.info(f"**下限**: ${conf_int_lower:.2f}")
            col2.success(f"**予測**: ${next_day_prediction:.2f}")
            col3.info(f"**上限**: ${conf_int_upper:.2f}")
            
            # 予測可視化
            st.markdown("---")
            st.subheader("📉 予測の可視化")
            
            recent_30 = data.tail(30)
            
            fig_forecast = go.Figure()
            
            # 過去データ
            fig_forecast.add_trace(go.Scatter(
                x=recent_30.index,
                y=recent_30.values,
                mode='lines+markers',
                name='実績',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            # 予測点
            fig_forecast.add_trace(go.Scatter(
                x=[next_day],
                y=[next_day_prediction],
                mode='markers',
                name='予測',
                marker=dict(size=15, color='red', symbol='star'),
            ))
            
            # エラーバー
            fig_forecast.add_trace(go.Scatter(
                x=[next_day],
                y=[next_day_prediction],
                mode='markers',
                marker=dict(size=15, color='red', opacity=0),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[conf_int_upper - next_day_prediction],
                    arrayminus=[next_day_prediction - conf_int_lower],
                    color='rgba(255,0,0,0.3)',
                    thickness=3,
                    width=10
                ),
                showlegend=False,
                name='95%信頼区間'
            ))
            
            fig_forecast.update_layout(
                xaxis_title="日付",
                yaxis_title="価格 (USD)",
                hovermode='x unified',
                height=450
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # モデル詳細
            st.markdown("---")
            st.subheader("🔍 モデル詳細")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**最適パラメータ**: ARIMA({p_opt}, {d_opt}, {q_opt})")
                st.info(f"**AIC**: {best_model.aic:.2f}")
                st.info(f"**BIC**: {best_model.bic:.2f}")
            
            with col2:
                st.info(f"**学習データ数**: {len(data)}日")
                st.info(f"**データ期間**: {data.index[0].strftime('%Y-%m-%d')} 〜 {data.index[-1].strftime('%Y-%m-%d')}")
                st.info(f"**予測信頼度**: 95%")
            
            # パラメータ説明
            with st.expander("📖 モデル詳細情報"):
                st.markdown(f"""
                ### ARIMA({p_opt}, {d_opt}, {q_opt})
                
                - **p = {p_opt}**: 自己回帰項（過去{p_opt}日の価格を考慮）
                - **d = {d_opt}**: 階差（{d_opt}回差分で定常化）
                - **q = {q_opt}**: 移動平均項（過去{q_opt}日の誤差を考慮）
                
                ### モデル評価指標
                
                - **AIC**: {best_model.aic:.2f} （小さいほど良い）
                - **BIC**: {best_model.bic:.2f} （小さいほど良い）
                """)
            
            with st.expander("📊 統計サマリー"):
                st.text(best_model.summary())
            
            # モデル適合度
            st.markdown("---")
            st.subheader("📈 モデルの適合度")
            
            fitted_values = best_model.fittedvalues
            recent_fit = pd.DataFrame({
                '実績': data.tail(30).values,
                'フィット': fitted_values[-30:]
            }, index=data.tail(30).index)
            
            fig_fit = go.Figure()
            fig_fit.add_trace(go.Scatter(
                x=recent_fit.index,
                y=recent_fit['実績'],
                mode='lines',
                name='実績',
                line=dict(color='blue', width=2)
            ))
            fig_fit.add_trace(go.Scatter(
                x=recent_fit.index,
                y=recent_fit['フィット'],
                mode='lines',
                name='モデル予測',
                line=dict(color='orange', width=2, dash='dash')
            ))
            fig_fit.update_layout(
                title="過去30日のモデル適合度",
                xaxis_title="日付",
                yaxis_title="価格 (USD)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_fit, use_container_width=True)
            
            # 残差統計
            residuals = best_model.resid
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("残差平均", f"{residuals.mean():.4f}")
            col2.metric("残差標準偏差", f"{residuals.std():.4f}")
            col3.metric("MAE", f"${np.mean(np.abs(residuals)):.2f}")
            col4.metric("RMSE", f"${np.sqrt(np.mean(residuals**2)):.2f}")
            
            st.success("✅ 分析完了!")
            st.warning("⚠️ **免責事項**: この予測は統計的手法に基づくものであり、実際の市場動向を保証するものではありません。")
            
        except Exception as e:
            st.error(f"モデル学習エラー: {str(e)}")
            st.info("💡 データ期間を変更するか、詳細設定を調整してください")

else:
    # 初期画面
    if api_key:
        st.info("👈 左サイドバーの「翌日価格を予測」ボタンをクリックしてください")
        
        st.markdown("""
        ## 📖 使い方
        
        1. **API Key確認**: 左サイドバーでAPI Keyが入力されていることを確認
        2. **データ設定**: データ取得量を選択（1ヶ月推奨）
        3. **予測実行**: 「翌日価格を予測」ボタンをクリック
        4. **結果確認**: 予測価格、信頼区間、モデル詳細を確認
        
        ## 💡 このアプリについて
        
        - **データ**: Alpha Vantage API経由でSPY（S&P500 ETF）のデータを取得
        - **モデル**: ARIMA（自動パラメータ最適化）
        - **予測**: 翌営業日の終値 + 95%信頼区間
        
        ### SPY (S&P500 ETF) とは？
        
        SPYはS&P500指数を追跡するETF（上場投資信託）で、S&P500の値動きとほぼ同じです。
        Alpha VantageではSPYのデータが最も信頼性が高いため、このアプリではSPYを使用しています。
        
        ### ⚠️ 注意事項
        
        - 無料プランは1分に5リクエストまで
        - 株価予測は統計的推定であり、確実性はありません
        - 投資判断は自己責任で行ってください
        """)

# フッター
st.markdown("---")
st.markdown("**📝 免責事項**: 教育・研究目的のみ | **🔬 データ**: Alpha Vantage API | **📊 モデル**: ARIMA (statsmodels)")



# サイドバーに追加
st.sidebar.markdown("---")
if st.sidebar.button("プライバシーポリシー"):
    st.markdown("""
    # プライバシーポリシー
    
    最終更新日: 2025年11月1日
    
    ## 広告について
    当サイトでは、第三者配信の広告サービス（Google AdSense）を利用しています。
    広告配信事業者は、ユーザーの興味に応じた広告を表示するためにCookieを使用することがあります。
    
    ## 個人情報の収集
    当サイトでは、ユーザーから個人情報を直接収集することはありません。
    Alpha Vantage APIキーはセッション中のみ使用され、保存されません。
    
    ## Cookieについて
    当サイトおよび第三者配信事業者は、Cookieを使用してユーザーの訪問履歴に基づいた広告を配信します。
    
    Cookieを無効にする方法については、
    [Google広告のポリシー](https://policies.google.com/technologies/ads?hl=ja)
    をご確認ください。
    
    ## お問い合わせ
    プライバシーポリシーに関するご質問は、以下までご連絡ください：
    メール: your-email@example.com
    
    ## 免責事項
    当サイトで提供する株価予測は、統計的手法に基づくものであり、
    投資助言や金融商品の勧誘ではありません。
    投資判断はご自身の責任で行ってください。
    """)