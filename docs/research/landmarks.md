# mediapipe landmark pose v1 & v2 비교

| 번호 | 이름 (`PoseLandmark`) | v1 포함 여부 | v2 포함 여부 | 비고 |
| --- | --- | --- | --- | --- |
| 0 | `NOSE` | ✅ | ✅ | 코 |
| 1 | `LEFT_EYE_INNER` | ✅ | ✅ | 왼쪽 눈 안쪽 |
| 2 | `LEFT_EYE` | ✅ | ✅ | 왼쪽 눈 |
| 3 | `LEFT_EYE_OUTER` | ✅ | ✅ | 왼쪽 눈 바깥쪽 |
| 4 | `RIGHT_EYE_INNER` | ✅ | ✅ | 오른쪽 눈 안쪽 |
| 5 | `RIGHT_EYE` | ✅ | ✅ | 오른쪽 눈 |
| 6 | `RIGHT_EYE_OUTER` | ✅ | ✅ | 오른쪽 눈 바깥쪽 |
| 7 | `LEFT_EAR` | ✅ | ✅ | 왼쪽 귀 |
| 8 | `RIGHT_EAR` | ✅ | ✅ | 오른쪽 귀 |
| 9 | `MOUTH_LEFT` | ✅ | ✅ | 입 왼쪽 |
| 10 | `MOUTH_RIGHT` | ✅ | ✅ | 입 오른쪽 |
| 11 | `LEFT_SHOULDER` | ✅ | ✅ | 왼쪽 어깨 |
| 12 | `RIGHT_SHOULDER` | ✅ | ✅ | 오른쪽 어깨 |
| 13 | `LEFT_ELBOW` | ✅ | ✅ | 왼쪽 팔꿈치 |
| 14 | `RIGHT_ELBOW` | ✅ | ✅ | 오른쪽 팔꿈치 |
| 15 | `LEFT_WRIST` | ✅ | ✅ | 왼쪽 손목 |
| 16 | `RIGHT_WRIST` | ✅ | ✅ | 오른쪽 손목 |
| 17 | `LEFT_PINKY` | ✅ | ✅ | 왼손 새끼손가락 |
| 18 | `RIGHT_PINKY` | ✅ | ✅ | 오른손 새끼손가락 |
| 19 | `LEFT_INDEX` | ✅ | ✅ | 왼손 검지 |
| 20 | `RIGHT_INDEX` | ✅ | ✅ | 오른손 검지 |
| 21 | `LEFT_THUMB` | ✅ | ✅ | 왼손 엄지 |
| 22 | `RIGHT_THUMB` | ✅ | ✅ | 오른손 엄지 |
| 23 | `LEFT_HIP` | ✅ | ✅ | 왼쪽 엉덩이 (고관절) |
| 24 | `RIGHT_HIP` | ✅ | ✅ | 오른쪽 엉덩이 (고관절) |
| 25 | `LEFT_KNEE` | ✅ | ✅ | 왼쪽 무릎 |
| 26 | `RIGHT_KNEE` | ✅ | ✅ | 오른쪽 무릎 |
| 27 | `LEFT_ANKLE` | ✅ | ✅ | 왼쪽 발목 |
| 28 | `RIGHT_ANKLE` | ✅ | ✅ | 오른쪽 발목 |
| 29 | `LEFT_HEEL` | ✅ | ✅ | 왼쪽 뒤꿈치 |
| 30 | `RIGHT_HEEL` | ✅ | ✅ | 오른쪽 뒤꿈치 |
| 31 | `LEFT_FOOT_INDEX` | ✅ | ✅ | 왼발 앞쪽 (엄지발가락 쪽) |
| 32 | `RIGHT_FOOT_INDEX` | ✅ | ✅ | 오른발 앞쪽 (엄지발가락 쪽) |
| 33 | `LEFT_FOREFOOT` | ❌ | ✅ | 왼발 전족부 (v2 추가) |
| 34 | `RIGHT_FOREFOOT` | ❌ | ✅ | 오른발 전족부 (v2 추가) |
| 35 | `LEFT_MIDHIP` | ❌ | ✅ | 왼쪽 골반 중간점 (v2 추가) |
| 36 | `RIGHT_MIDHIP` | ❌ | ✅ | 오른쪽 골반 중간점 (v2 추가) |
| 37 | `MIDHIP` | ❌ | ✅ | 좌우 골반 중간 중심점 (v2 추가) |

- Pose v1: 총 33개 랜드마크
- Pose v2: 총 38개 랜드마크
- v2 추가 포인트: 골반 중심(MIDHIP) 및 발 앞쪽(FOREFOOT) 관련 포인트가 정렬·균형 분석용으로 추가됨
- 즉, v2는 하체 균형 및 보행 정렬 분석에 더 정밀한 분석이 가능