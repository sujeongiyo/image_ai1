import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

# 페이지 설정
st.set_page_config(
    page_title="필기체 숫자 인식기",
    page_icon="✏️",
    layout="centered"
)

# 제목 및 설명 추가
st.title("필기체 숫자 인식기")
st.markdown("손으로 쓴 숫자 이미지를 업로드하면 AI가 어떤 숫자인지 예측합니다.")

@st.cache_resource
def load_mnist_model():
    """사전 훈련된 MNIST 모델을 로드합니다."""
    try:
        model = load_model("mnist_classifier.h5")
        return model
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None

def preprocess_image(image):
    """이미지를 MNIST 모델 입력 형식에 맞게 전처리합니다."""
    # 이미지를 흑백으로 변환
    image = image.convert('L')
    # 28x28 크기로 조정
    image = image.resize((28, 28))
    # 이미지를 numpy 배열로 변환
    img_array = np.array(image)
    # 픽셀값 반전 (MNIST 데이터셋은 검은 배경에 흰색 숫자)
    img_array = 255 - img_array
    # 정규화 (0-1 범위로)
    img_array = img_array / 255.0
    # 모델 입력 형태로 변환
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

# 모델 로드
model = load_mnist_model()

# 파일 업로더 위젯 추가
uploaded_file = st.file_uploader("필기체 숫자 이미지 업로드", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 표시
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="업로드된 이미지", width=200)
    
    # 이미지 전처리 및 예측
    processed_img = preprocess_image(image)
    
    with col2:
        st.image(processed_img.reshape(28, 28), caption="전처리된 이미지", width=150)
    
    if model is not None:
        # 예측 버튼
        if st.button("숫자 예측하기"):
            with st.spinner('예측 중...'):
                # 예측 실행
                prediction = model.predict(processed_img)
                predicted_digit = np.argmax(prediction[0])
                confidence = prediction[0][predicted_digit] * 100
                
                # 결과 표시
                st.success(f"예측 결과: **{predicted_digit}**")
                
                st.write(f"확신도: {confidence:.2f}%")
                
                # 전체 예측 분포 시각화
                st.bar_chart(prediction[0])
    else:
        st.error("모델을 로드할 수 없습니다. 모델 파일이 올바른 경로에 있는지 확인하세요.")

# 사용 방법 설명
with st.expander("사용 방법"):
    st.markdown("""
    1. '필기체 숫자 이미지 업로드' 버튼을 클릭하여 손으로 쓴 숫자 이미지를 선택합니다.
    2. 이미지는 jpg, jpeg, png 형식이어야 합니다.
    3. '숫자 예측하기' 버튼을 클릭하면 AI가 이미지에서 어떤 숫자인지 예측합니다.
    4. 깨끗한 흰색 배경에 검은색으로 숫자를 쓴 이미지가 가장 잘 인식됩니다.
    """)

# 푸터 추가
st.markdown("---")
st.markdown("MNIST 데이터셋으로 훈련된 딥러닝 모델을 사용한 필기체 숫자 인식기")
