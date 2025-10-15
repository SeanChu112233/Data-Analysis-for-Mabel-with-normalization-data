import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, ShuffleSplit
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 页面配置
st.set_page_config(
    page_title="推出力-温度/时间条件数据分析工具 for Mabel",
    page_icon="📊",
    layout="wide"
)

# 标题和说明
st.title("📊 推出力-温度/时间条件数据分析工具 for Mabel")
st.markdown("""
    该工具用于分析温度(temp)、时间(time)与推出力(force)之间的关系，
    支持多种回归模型分析，并提供可视化结果。程序会自动对比数据归一化前后的模型表现。
    请上传包含相关数据的CSV文件。
""")

# 侧边栏 - 文件上传和参数设置
with st.sidebar:
    st.header("设置")
    
    # 文件上传
    uploaded_file = st.file_uploader("上传CSV数据文件", type=["csv"])
    
    # 模型选择
    model_option = st.selectbox(
        "选择回归模型",
        ("随机森林", "梯度提升树", "多项式回归", "线性回归")
    )
    
    # 显示数据信息
    st.subheader("数据信息")
    data_info = st.empty()

# 主程序
def main():
    # 检查是否上传了文件
    if uploaded_file is not None:
        try:
            # 读取CSV文件
            data = pd.read_csv(uploaded_file)
            
            # 检查必要的列是否存在
            required_columns = ['temp', 'time', 'force']
            data.columns = [col.lower() for col in data.columns]  # 转换列名为小写
            missing = [col for col in required_columns if col not in data.columns]
            
            if missing:
                st.error(f"数据文件缺少必要的列: {', '.join(missing)}")
                return
            
            # 重命名列名，首字母大写
            data = data.rename(columns={
                'temp': 'Temp',
                'time': 'Time',
                'force': 'Force'
            })
            
            # 在侧边栏显示数据信息
            with st.sidebar:
                data_info.dataframe(data.describe(), use_container_width=True)
                st.write(f"数据量: {len(data)} 条")
            
            # 显示原始数据
            with st.expander("查看原始数据", expanded=False):
                st.dataframe(data, use_container_width=True)
            
            # 准备数据 - 原始和归一化版本
            X_original = data[['Temp', 'Time']]
            y = data['Force']
            
            # 数据归一化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_original)
            X_scaled_df = pd.DataFrame(X_scaled, columns=['Temp', 'Time'])
            
            # 显示归一化前后的数据对比
            with st.expander("查看归一化前后数据对比", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("原始数据")
                    st.dataframe(X_original.describe(), use_container_width=True)
                with col2:
                    st.subheader("归一化后数据")
                    st.dataframe(X_scaled_df.describe(), use_container_width=True)
            
            # 训练模型 - 原始和归一化两种版本
            models_original = train_models(X_original, y)
            models_scaled = train_models(X_scaled_df, y)
            
            # 对比归一化前后的模型性能
            compare_normalization_effect(models_original, models_scaled, model_option)
            
            # 选择更优的模型集
            best_version, best_models = select_better_version(models_original, models_scaled, model_option)
            st.success(f"基于交叉验证R²值，选择{'归一化后' if best_version == 'scaled' else '原始数据'}的模型进行后续分析")
            
            # 显示模型评估结果
            show_model_evaluation(best_models, data, best_version)
            
            # 显示可视化结果
            show_visualizations(best_models, data, model_option, best_version, scaler if best_version == 'scaled' else None)
            
            # 交互式预测
            show_prediction_tool(best_models, data, model_option, best_version, scaler if best_version == 'scaled' else None)
            
        except Exception as e:
            st.error(f"处理数据时出错: {str(e)}")
    else:
        # 显示示例数据
        st.info("请上传CSV文件开始分析。示例数据格式如下：")
        sample_data = pd.DataFrame({
            'temp': [70, 70, 70, 90, 90],
            'time': [60, 180, 300, 60, 180],
            'force': [3.4, 3.1, 3.3, 3.1, 2.8]
        })
        st.dataframe(sample_data, use_container_width=True)

def train_models(X, y):
    """训练所有回归模型"""
    models = {}
    
    # 1. 线性回归
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    models["线性回归"] = {
        "model": linear_model,
        "type": "linear"
    }
    
    # 2. 多项式回归 (二次项)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y)
    models["多项式回归"] = {
        "model": poly_model,
        "type": "polynomial",
        "poly_transform": poly
    }
    
    # 3. 随机森林回归
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    models["随机森林"] = {
        "model": rf_model,
        "type": "nonlinear"
    }
    
    # 4. 梯度提升树回归
    gb_model = GradientBoostingRegressor(
        n_estimators=30,        # 减少树的数量
        max_depth=3,            # 限制树深度
        min_samples_leaf=2,     # 增加叶节点最小样本数
        subsample=0.8,          # 使用80%的样本训练每棵树
        random_state=42
    )
    gb_model.fit(X, y)
    models["梯度提升树"] = {
        "model": gb_model,
        "type": "nonlinear"
    }
    
    # 评估模型
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    for name, model_info in models.items():
        model = model_info["model"]
        
        if model_info["type"] == "polynomial":
            X_processed = model_info["poly_transform"].transform(X)
        else:
            X_processed = X
        
        # 计算交叉验证得分
        cv_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='r2')
        cv_rmse = cross_val_score(model, X_processed, y, cv=cv, scoring='neg_mean_squared_error')
        
        # 计算训练R²和RMSE
        y_pred = predict_with_model(name, models, X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        model_info["cv_r2_mean"] = np.mean(cv_scores)
        model_info["cv_r2_std"] = np.std(cv_scores)
        model_info["cv_rmse_mean"] = np.mean(np.sqrt(-cv_rmse))
        model_info["train_r2"] = r2
        model_info["train_rmse"] = rmse
    
    return models

def predict_with_model(model_name, models, X):
    """使用指定模型进行预测"""
    model_info = models[model_name]
    model = model_info["model"]
    
    if model_info["type"] == "polynomial":
        X_processed = model_info["poly_transform"].transform(X)
        return model.predict(X_processed)
    else:
        return model.predict(X)

def compare_normalization_effect(models_original, models_scaled, selected_model):
    """对比归一化前后的模型性能"""
    st.subheader("📊 归一化前后模型性能对比")
    
    # 准备对比数据
    compare_data = []
    
    # 对比所选模型
    orig_info = models_original[selected_model]
    scaled_info = models_scaled[selected_model]
    
    compare_data.append({
        "评估指标": "训练R²",
        "原始数据": f"{orig_info['train_r2']:.4f}",
        "归一化后": f"{scaled_info['train_r2']:.4f}",
        "变化": f"{(scaled_info['train_r2'] - orig_info['train_r2']):+.4f}"
    })
    
    compare_data.append({
        "评估指标": "交叉验证R²",
        "原始数据": f"{orig_info['cv_r2_mean']:.4f}",
        "归一化后": f"{scaled_info['cv_r2_mean']:.4f}",
        "变化": f"{(scaled_info['cv_r2_mean'] - orig_info['cv_r2_mean']):+.4f}"
    })
    
    compare_data.append({
        "评估指标": "训练RMSE",
        "原始数据": f"{orig_info['train_rmse']:.4f}",
        "归一化后": f"{scaled_info['train_rmse']:.4f}",
        "变化": f"{(scaled_info['train_rmse'] - orig_info['train_rmse']):+.4f}"
    })
    
    compare_df = pd.DataFrame(compare_data)
    
    # 高亮显示更好的结果
    def highlight_better(row):
        styles = []
        metric = row['评估指标']
        orig_val = float(row['原始数据'])
        scaled_val = float(row['归一化后'])
        
        for col in row.index:
            if col == '原始数据':
                # R²越大越好，RMSE越小越好
                if (metric.endswith('R²') and orig_val > scaled_val) or \
                   (metric.endswith('RMSE') and orig_val < scaled_val):
                    styles.append('background-color: #90EE90')
                else:
                    styles.append('')
            elif col == '归一化后':
                if (metric.endswith('R²') and scaled_val > orig_val) or \
                   (metric.endswith('RMSE') and scaled_val < orig_val):
                    styles.append('background-color: #90EE90')
                else:
                    styles.append('')
            else:
                styles.append('')
        return styles
    
    styled_df = compare_df.style.apply(highlight_better, axis=1)
    st.dataframe(styled_df, use_container_width=True)

def select_better_version(models_original, models_scaled, selected_model):
    """选择性能更好的模型版本（原始或归一化）"""
    orig_score = models_original[selected_model]['cv_r2_mean']
    scaled_score = models_scaled[selected_model]['cv_r2_mean']
    
    # 如果归一化后交叉验证R²更高，则选择归一化版本
    if scaled_score > orig_score:
        return 'scaled', models_scaled
    else:
        return 'original', models_original

def show_model_evaluation(models, data, version):
    """显示模型评估结果"""
    st.subheader(f"📈 模型评估结果 ({'归一化后' if version == 'scaled' else '原始数据'})")
    
    # 准备评估结果数据
    eval_data = []
    for name, info in models.items():
        eval_data.append({
            "模型": name,
            "训练R²": f"{info['train_r2']:.4f}",
            "交叉验证R²": f"{info['cv_r2_mean']:.4f} ± {info['cv_r2_std']:.4f}",
            "训练RMSE": f"{info['train_rmse']:.4f}"
        })
    
    eval_df = pd.DataFrame(eval_data)
    
    # 找出最佳模型（交叉验证R²最高）
    best_model = max(models.items(), key=lambda x: x[1]["cv_r2_mean"])[0]
    
    # 显示评估表格，突出最佳模型
    def highlight_best(row):
        return ['background-color: #90EE90' if row['模型'] == best_model else '' for _ in row]
    
    styled_df = eval_df.style.apply(highlight_best, axis=1)
    st.dataframe(styled_df, use_container_width=True)
    
    st.info(f"推荐模型: **{best_model}** (基于交叉验证R²最高)")

def show_visualizations(models, data, model_name, version, scaler):
    """显示数据可视化结果"""
    st.subheader("🔍 数据可视化")
    
    # 创建网格数据用于绘制曲面
    x = data['Temp']
    y = data['Time']
    z = data['Force']
    
    x_range = np.linspace(x.min(), x.max(), 30)
    y_range = np.linspace(y.min(), y.max(), 30)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    grid_data = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    # 如果使用归一化模型，需要对网格数据进行归一化
    if version == 'scaled' and scaler is not None:
        grid_data_scaled = scaler.transform(grid_data)
        grid_data_processed = pd.DataFrame(grid_data_scaled, columns=['Temp', 'Time'])
    else:
        grid_data_processed = pd.DataFrame(grid_data, columns=['Temp', 'Time'])
    
    # 获取模型预测值
    Z_grid = predict_with_model(model_name, models, grid_data_processed)
    Z_grid = Z_grid.reshape(X_grid.shape)
    
    # 分为两列显示图表
    col1, col2 = st.columns(2)
    
    with col1:
        # 3D散点图和曲面图 (使用plotly)
        st.subheader(f"3D关系图 ({model_name})")
        
        # 创建3D图形
        fig = go.Figure()
        
        # 添加原始数据点
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=5, color='red'),
            name='实验数据'
        ))
        
        # 添加预测曲面
        fig.add_trace(go.Surface(
            x=X_grid, y=Y_grid, z=Z_grid,
            opacity=0.6,
            colorscale='Viridis',
            name='预测曲面'
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='温度 (°C)',
                yaxis_title='时间 (s)',
                zaxis_title='推出力'
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 热区图
        st.subheader(f"热区图 ({model_name})")
        
        # 创建热区图数据
        z_dense = Z_grid
        fig = px.imshow(
            z_dense,
            x=x_range,
            y=y_range,
            color_continuous_scale='Viridis',
            aspect='auto',
            labels=dict(x="温度 (°C)", y="时间 (s)", color="推出力")
        )
        
        # 添加原始数据点
        fig.add_scatter(x=x, y=y, mode='markers', 
                       marker=dict(color='red', size=8, line=dict(width=2, color='black')),
                       name='实验数据')
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # 相关性分析
    st.subheader("📊 相关性分析")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def show_prediction_tool(models, data, model_name, version, scaler):
    """显示交互式预测工具"""
    st.subheader("🔮 推出力预测")
    
    # 获取数据范围
    temp_min, temp_max = data['Temp'].min(), data['Temp'].max()
    time_min, time_max = data['Time'].min(), data['Time'].max()
    
    # 添加一些缓冲
    temp_buffer = (temp_max - temp_min) * 0.1
    time_buffer = (time_max - time_min) * 0.1
    
    # 创建输入控件
    col1, col2 = st.columns(2)
    
    with col1:
        temp = st.slider(
            "温度 (°C)",
            min_value=float(temp_min - temp_buffer),
            max_value=float(temp_max + temp_buffer),
            value=float(data['Temp'].mean())
        )
    
    with col2:
        time = st.slider(
            "时间 (s)",
            min_value=float(time_min - time_buffer),
            max_value=float(time_max + time_buffer),
            value=float(data['Time'].mean())
        )
    
    # 准备预测数据
    X_pred = pd.DataFrame([[temp, time]], columns=['Temp', 'Time'])
    
    # 如果使用归一化模型，需要对输入数据进行归一化
    if version == 'scaled' and scaler is not None:
        X_pred_scaled = scaler.transform(X_pred)
        X_pred_processed = pd.DataFrame(X_pred_scaled, columns=['Temp', 'Time'])
    else:
        X_pred_processed = X_pred
    
    # 预测推出力
    force_pred = predict_with_model(model_name, models, X_pred_processed)[0]
    
    # 显示预测结果
    st.metric("预测推出力", f"{force_pred:.4f}", delta=None)
    
    # 显示模型影响分析
    st.subheader("📋 影响分析")
    model_info = models[model_name]
    
    if model_name == "线性回归":
        coefs = model_info["model"].coef_
        intercept = model_info["model"].intercept_
        
        if version == 'scaled':
            st.info("注意：归一化后的线性回归系数不能直接比较原始特征的影响大小，因为特征已被标准化")
            st.latex(f"推出力 = {intercept:.4f} + {coefs[0]:.4f} \\times 标准化温度 + {coefs[1]:.4f} \\times 标准化时间")
        else:
            st.latex(f"推出力 = {intercept:.4f} + {coefs[0]:.4f} \\times 温度 + {coefs[1]:.4f} \\times 时间")
        
        st.write(f"决定系数 R²: {model_info['train_r2']:.4f}")
        
        # 分析影响大小（仅对原始数据有意义）
        if version != 'scaled':
            temp_impact = abs(coefs[0])
            time_impact = abs(coefs[1])
            
            if temp_impact > time_impact:
                st.info(f"温度对推出力的影响更大 (影响比例: {temp_impact/time_impact:.2f}:1)")
            elif time_impact > temp_impact:
                st.info(f"时间对推出力的影响更大 (影响比例: 1:{time_impact/temp_impact:.2f})")
            else:
                st.info("温度和时间对推出力的影响大致相同")
    
    elif model_name == "多项式回归":
        st.write(f"决定系数 R²: {model_info['train_r2']:.4f}")
        if version == 'scaled':
            st.info("模型包含标准化温度、标准化时间的二次项以及交互项，表明它们对推出力的影响是非线性的")
        else:
            st.info("模型包含温度、时间的二次项以及交互项，表明它们对推出力的影响是非线性的，在不同范围内影响程度不同")
    
    else:  # 随机森林和梯度提升树
        st.write(f"决定系数 R²: {model_info['train_r2']:.4f}")
        
        # 特征重要性
        importances = model_info["model"].feature_importances_
        temp_importance = importances[0]
        time_importance = importances[1]
        
        st.write(f"温度特征重要性: {temp_importance:.4f}")
        st.write(f"时间特征重要性: {time_importance:.4f}")
        
        if temp_importance > time_importance:
            st.info(f"温度对推出力的影响更大 (重要性比例: {temp_importance/time_importance:.2f}:1)")
        elif time_importance > temp_importance:
            st.info(f"时间对推出力的影响更大 (重要性比例: 1:{time_importance/temp_importance:.2f})")
        else:
            st.info("温度和时间对推出力的影响大致相同")

# 添加页脚信息
def add_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center;">
        <p>团队贡献数据分析工具 | 数据来源于上传的CSV文件</p>
        <p>© 2023 团队贡献分析项目</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    add_footer()
