# project_hj_final Workspace README

## 1) 개요

이 워크스페이스는 ROS 2 기반으로 아래 4가지 축을 중심으로 구성됩니다.

1. **Intention 추정/정합**
   - `src/intention/intention/intention_transformer.py` (핵심)
2. **테스트용 가짜 박스 퍼블리셔**
   - `src/intention/intention/fake_box_pose_publisher.py` (실사용 목적 아님)
3. **UR 로봇 텔레오퍼레이션 제어**
   - `src/robot_control/robot_control/quest_teleop_base.py`
4. **Unity 연동 PointCloud 표시**
   - `src/unity_ros_client/unity_ros_client/pcd_service.py`

또한 PointCloud 관련 커스텀 서비스 인터페이스(`unity_ros_interfaces/srv/PcdService.srv`)와 다수의 서드파티 패키지(apriltag, realsense, rtabmap 등)가 함께 포함되어 있습니다.

---

## 2) 필수 코드 설명

---

### 2.1 `fake_box_pose_publisher.py`  
> 경로: `src/intention/intention/fake_box_pose_publisher.py`  
> **용도: 테스트/디버깅용 (실제 운용용 아님)**

#### 역할
- `virtual_box`(map frame 기준)와 `real_box`(base_link 기준) MarkerArray를 생성/퍼블리시합니다.
- 미리 정의된 박스 시드(`VIRTUAL_BOX_SEEDS`)를 사용하며, 가우시안 노이즈를 추가하여 실제 박스를 흉내냅니다.
- `map -> base_link` TF를 조회해 virtual→real 변환을 수행합니다.

#### 핵심 포인트
- TF가 없을 때 identity fallback 옵션이 있으나, 기본값은 `False`로 실제 통합 테스트에서 TF 문제를 드러내도록 되어 있습니다.
- 위치/자세 노이즈 표준편차 및 고정/가변 노이즈 모드를 코드 상수로 관리합니다.
- **intention_transformer 입력 토픽을 인위적으로 만들어주는 도구**로 보는 것이 정확합니다.

#### 언제 쓰는가?
- SLAM/인식 없이 intention 알고리즘만 독립 검증할 때
- `/virtual_box`, `/real_box` 파이프라인/시각화 확인할 때

---

### 2.2 `intention_transformer.py` (코어)
> 경로: `src/intention/intention/intention_transformer.py`  
> **이 워크스페이스의 핵심 로직**

#### 역할
- ROS1의 분리된 intention/pose 관련 노드들을 ROS2 단일 파일로 통합한 구현입니다.
- 입력:
  - `/real_box`, `/virtual_box` (`MarkerArray`)
  - `/joint_states`
  - `/controller/right/joy` (리셋 버튼)
  - TF (`map`, `base_link`, `tool0`)
- 출력:
  - `/best_box/selected` (최적 real/virtual box MarkerArray)
  - `/intention/debug_markers`
  - `map -> virtual_base_link` TF

#### 알고리즘 개요
1. tool0 움직임(속도/방향)을 TF로 추정
2. 박스/평면 정보와 결합해 가우시안 기반 intention score 계산
3. best box pair를 선택
4. 선택된 real/virtual box 정합으로 `virtual_base_link`를 갱신
   - orientation 포함 정합 또는 위치-only 정합(옵션 `IGNORE_ORIENTATION`)

#### 특징
- plane은 고정 파라미터 또는 real box 기반 회귀(PCA)로 계산 가능
- 디버그 marker(평면/교차점 등) 지원
- 파라미터 서버 대신 파일 상수 중심으로 제어 (빠른 실험 지향)

---

### 2.3 `quest_teleop_base.py`
> 경로: `src/robot_control/robot_control/quest_teleop_base.py`  
> **실제 UR 로봇 텔레오퍼레이션 제어 코드**

#### 역할
- Quest 컨트롤러(`/quest/right/joy`, `/quest/right/pose`)를 읽어 UR RTDE로 `servoL` 제어를 수행합니다.
- grip 버튼으로 tracking 시작/종료.
- 시작 시점 pose를 기준으로 상대 이동량/회전량을 계산합니다.

#### 핵심 제어/안전 로직
- `ur5e <- xr_origin` TF를 조회해 XR 좌표계를 로봇 기준으로 변환
- 180도 Z 보정 매트릭스 적용 (환경 설정 이슈 보정)
- IK 해 존재 여부, 관절 안전 한계, 특이점/충돌 위험, 해 점프(q_diff) 등을 다단계로 필터링
- low-pass filter(EMA) 후 `servoL` 명령

#### 같은 경로의 다른 제어 코드와 비교

##### A) `quest_teleop_world.py` 대비
- `world.py`: 컨트롤러 이동량을 **base/world 기준**으로 직접 적용하는 형태
- `base.py`: TF(`ur5e <- xr_origin`)를 매 주기 반영하여 **XR 기준 변화가 로봇 기준에 맞게 회전 변환**됨
- 결과적으로 `base.py`는 모바일 베이스 이동/시점 변화가 있는 환경에서 상대적으로 일관된 조작감을 기대할 수 있음

##### B) `quest_teleop_EE.py` 대비
- `EE.py`: 컨트롤러 이동을 초기 EE 기준 회전(`R_robot_init`)에 투영해 적용 (EE frame 지향)
- `base.py`: XR 원점과 UR base 간 TF를 직접 사용해 좌표계 매핑
- `EE.py`는 초기 툴자세 기준 조작감, `base.py`는 XR 공간 기준 조작감에 가까움

##### C) 공통점
- 세 파일 모두 grip 기반 tracking on/off, IK 안전검사, `servoL` 기반 주기 제어 구조는 유사
- 차이는 **이동/회전 델타를 어떤 좌표계에서 해석하느냐**에 집중됨

---

### 2.4 `pcd_service.py`
> 경로: `src/unity_ros_client/unity_ros_client/pcd_service.py`  
> **PointCloud Display용 서비스형 퍼블리셔**

#### 역할
- Open3D로 PCD를 로드하고 다운샘플링한 뒤 `sensor_msgs/PointCloud2`로 변환합니다.
- 평소엔 대기하고, `Trigger` 서비스(`/publish_pcd_trigger`) 호출 시 1회 publish 합니다.
- 퍼블리시 토픽: `pcd_cloud` (Unity Subscriber 대상)

#### 구현 포인트
- 시작 시 메시지를 미리 준비하여 서비스 호출 지연 최소화
- RGB를 packed float32로 PointCloud2 필드에 인코딩
- frame_id는 `map` 사용
- 기본 pcd 경로 리스트에서 index를 선택하는 방식(현재 코드상 하드코딩)

---

## 3) 기타 중요 구성요소 (권장 포함)

### 3.1 `unity_ros_interfaces/srv/PcdService.srv`
- `trigger`, `index`, `sampling_value`를 받는 커스텀 서비스 정의입니다.
- 현재 `pcd_service.py`는 표준 `std_srvs/Trigger`를 사용하지만, 같은 패키지의 `pcd_custom_service.py`에서 커스텀 인터페이스 확장에 활용할 수 있습니다.

### 3.2 패키지 엔트리포인트/설치 관점
- `unity_ros_client`는 `pcd_service`, `pcd_custom_service`, `cmd_publisher*`를 콘솔 스크립트로 노출
- `robot_control`도 여러 teleop/test 엔트리를 정의
- 반면 `intention` 패키지는 현재 `setup.py`의 `console_scripts`가 비어 있어 직접 실행 방식(또는 향후 엔트리 추가)이 필요합니다.

### 3.3 대형 서브트리
- `realsense-ros-r-4.56.4`, `rtabmap_ros`, `rtabmap`, `apriltag`, `serial` 등 외부/기반 패키지가 공존합니다.
- 본 README는 사용자 요청 범위상 **intention/robot_control/unity_ros_client 중심**으로 정리했습니다.

---

## 4) 권장 실행 흐름(개발/디버그)

1. 로봇 없이 intention 로직 검증  
   - `fake_box_pose_publisher`로 `/virtual_box`, `/real_box` 생성
   - `intention_transformer`에서 best box/가상 base TF 확인
2. Unity 맵 시각화
   - `pcd_service` 실행 후 `/publish_pcd_trigger` 호출해 pointcloud 1회 송신
3. 실제 로봇 텔레옵
   - `quest_teleop_base.py` 기반으로 XR↔UR TF가 정상인지 먼저 점검 후 tracking 시작

---

## 5) 주의사항

- 코드 전반에 하드코딩된 토픽/프레임/파일경로/파라미터가 많으므로 환경별 정리가 필요합니다.
- UR RTDE 제어는 네트워크/안전장치 상태에 민감하므로 드라이런 후 실제 운용하십시오.
- `intention_transformer.py`는 `rotutils` 의존성이 있으므로 실행 환경에 해당 모듈이 준비되어야 합니다.