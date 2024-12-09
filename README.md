## Overview
목표: 다국어 영수증에 적용 가능한 OCR 모델을 위한 데이터 전처리

데이터셋: 중국어, 일본어, 태국어, 베트남어 영수증 이미지

평가지표: DetEval 

<br><br>

## Member  
<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/kim-minsol"><img height="110px" src="https://avatars.githubusercontent.com/u/81224613?v=4"/></a>
            <br />
            <a href="https://github.com/kim-minsol"><strong>김민솔</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/joonhyunkim1"><img height="110px"  src="https://avatars.githubusercontent.com/u/141805564?v=4"/></a>
              <br />
              <a href="https://github.com/joonhyunkim1"><strong>김준현</strong></a>
              <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/sweetie-orange"><img height="110px"  src="https://avatars.githubusercontent.com/u/97962649?v=4"/></a>
              <br />
              <a href="https://github.com/sweetie-orange"><strong>김현진</strong></a>
              <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/0seoYun"><img height="110px"  src="https://avatars.githubusercontent.com/u/102219161?v=4"/></a>
              <br />
              <a href="https://github.com/0seoYun"><strong>윤영서</strong></a>
              <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/2JAE22"><img height="110px"  src="https://avatars.githubusercontent.com/u/87936538?v=4"/></a>
              <br />
              <a href="https://github.com/2JAE22"><strong>이재건</strong></a>
              <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/Gwonee"><img height="110px"  src="https://avatars.githubusercontent.com/u/125177607?v=4"/></a>
              <br />
              <a href="https://github.com/Gwonee"><strong>정권희</strong></a>
              <br />
        </td>
    </tr>
</table>  

<br>

### 역할

|팀원|역할|
|-----|---|
|김민솔| Dataset 추가 및 합성 |
|김준현| Labelling, 데이터 시각화 |
|김현진| Augmentation |
|윤영서| Dataset 추가 및 합성 |
|이재건| Labelling, 데이터 시각화 |
|정권희| Augmentation |

<br><br>


## Methods

|분류|내용|
  |-----|---|
  |Dataset|  **Labelling** <br> -orientation 수정 <br> -라벨 노이즈 제거 <br> **외부 데이터 추가** <br> - ICDAR 데이터셋 <br> - CORD 데이터셋|
  |Augmentation| - translate_image <br> - perspective_transform <br> - adjust_brightness_contrast_saturation <br> - add_gaussian_noise <br> - add_salt_and_pepper_noise <br> - overlay_text <br> -add_random_lines|
  |시각화 | Streamlit |

<br><br>

## Result
||Public Dataset|Private Dataset|
|---|-----|---|
|Precision| 0.8597 | 0.8342 |
|recall| 0.7656 | 0.7830 |
|f1| 0.8100 | 0.8078 |


