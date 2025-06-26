#!/bin/bash

# 공유 폴더 기본 경로
SHARE_DIR="/mnt/hgfs/vmware_share_file"

# 📌 1. VMware 공유 폴더 수동 마운트 시도
echo "🔄 공유 폴더 마운트 시도 중..."
sudo mount -t fuse.vmhgfs-fuse .host:/vmware_share_file "$SHARE_DIR" -o allow_other 2>/dev/null

if mount | grep -q "$SHARE_DIR"; then
  echo "✅ 공유 폴더 마운트 완료: $SHARE_DIR"
else
  echo "⚠️ 공유 폴더 마운트 실패. 이미 마운트되었거나 오류 발생."
fi

# 📌 1.5. 공유 폴더 파일 목록 출력
echo ""
echo "📂 현재 공유 폴더 내 파일 목록:"
ls -al --color=auto "$SHARE_DIR"

# 📌 2. 복사 방향 선택
echo ""
echo "📁 복사 방향을 선택하세요:"
echo "1. w2u (윈도우 → 우분투)"
echo "2. u2w (우분투 → 윈도우)"
read -p "입력 (w2u 또는 u2w): " DIRECTION

# 입력 검증
if [[ "$DIRECTION" != "w2u" && "$DIRECTION" != "u2w" ]]; then
  echo "❌ 잘못된 입력입니다. 'w2u' 또는 'u2w'만 입력하세요."
  exit 1
fi

# 📌 3. 조건별 입력 처리
if [[ "$DIRECTION" == "w2u" ]]; then
  read -p "📄 공유 폴더 내 파일 이름 (예: test.csv): " FILENAME
  SRC="$SHARE_DIR/$FILENAME"

  if [[ ! -f "$SRC" ]]; then
    echo "❌ 공유 폴더에서 파일을 찾을 수 없습니다: $SRC"
    exit 1
  fi

  read -p "📂 복사할 우분투 내 도착 경로 (절대경로): " DEST_PATH

  mkdir -p "$(dirname "$DEST_PATH")"  # 상위 폴더 없으면 생성
  cp "$SRC" "$DEST_PATH" && echo "✅ 복사 완료: $SRC → $DEST_PATH" || echo "❌ 복사 실패"

elif [[ "$DIRECTION" == "u2w" ]]; then
  read -p "📄 우분투 내 복사할 파일 절대경로: " SRC

  if [[ ! -f "$SRC" ]]; then
    echo "❌ 우분투 파일이 존재하지 않습니다: $SRC"
    exit 1
  fi

  FILENAME=$(basename "$SRC")
  DEST_PATH="$SHARE_DIR/$FILENAME"

  cp "$SRC" "$DEST_PATH" && echo "✅ 복사 완료: $SRC → $DEST_PATH" || echo "❌ 복사 실패"
fi

# 작성자 : umi