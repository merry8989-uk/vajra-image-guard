#!/usr/bin/env bash
#
# Build, align and sign the Vajra APK without Gradle.
#
#   ANDROID_SDK_ROOT=/path/to/android-sdk ./build-apk.sh
#
# Needs: JDK 17+ on PATH, Android SDK with build-tools 34.0.0 and platform 34.
# The first run generates a signing key in .keystore/ (git-ignored) and reuses it
# afterwards, so rebuilds upgrade cleanly over each other.
set -euo pipefail

# Containers that cap the address space (ulimit -v) starve the tool JVMs; d8 in
# particular wants room to breathe. Raising the soft limit is harmless elsewhere.
ulimit -v unlimited 2>/dev/null || true

cd "$(dirname "$0")"

SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
if [ -z "$SDK" ]; then
  echo "error: set ANDROID_SDK_ROOT (or ANDROID_HOME) to your Android SDK" >&2
  exit 1
fi

# build-tools may sit directly under the SDK or under a versioned folder.
BT=""
for candidate in "$SDK/build-tools/34.0.0" "$SDK/build-tools"; do
  if [ -x "$candidate/aapt2" ]; then BT="$candidate"; break; fi
done
if [ -z "$BT" ]; then
  BT="$(find "$SDK/build-tools" -maxdepth 2 -name aapt2 -type f 2>/dev/null | head -1 | xargs -r dirname)"
fi
[ -n "$BT" ] || { echo "error: aapt2 not found under $SDK/build-tools" >&2; exit 1; }

PLATFORM="$SDK/platforms/android-34/android.jar"
if [ ! -f "$PLATFORM" ]; then
  # fall back to whichever platform is installed, newest first
  PLATFORM="$(ls -d "$SDK"/platforms/android-*/android.jar 2>/dev/null | sort -V | tail -1)"
fi
[ -f "$PLATFORM" ] || { echo "error: no android.jar under $SDK/platforms (install a platform)" >&2; exit 1; }

# d8 defaults to a 2 GB heap, which is more than a small CI container has.
# Override with BUILD_HEAP=512M if the tool JVMs get killed.
HEAP="${BUILD_HEAP:-1G}"

VERSION_NAME="$(sed -n 's/.*android:versionName="\([^"]*\)".*/\1/p' app/src/main/AndroidManifest.xml | head -1)"
VERSION_CODE="$(sed -n 's/.*android:versionCode="\([^"]*\)".*/\1/p' app/src/main/AndroidManifest.xml | head -1)"
OUT="vajra-image-guard-${VERSION_NAME}.apk"

echo "==> staging web assets"
rm -rf app/src/main/assets/www
mkdir -p app/src/main/assets/www
cp -R ../docs/. app/src/main/assets/www/
# never package the published APK into itself, nor the UI test driver
rm -rf app/src/main/assets/www/download
rm -f app/src/main/assets/www/_selftest.html

echo "==> aapt2 compile"
rm -rf build
mkdir -p build/classes build/dex
"$BT/aapt2" compile --dir app/src/main/res -o build/res.zip

echo "==> aapt2 link"
"$BT/aapt2" link \
  -o build/unsigned.apk \
  -I "$PLATFORM" \
  --manifest app/src/main/AndroidManifest.xml \
  -A app/src/main/assets \
  --min-sdk-version 24 \
  --target-sdk-version 34 \
  --version-code "$VERSION_CODE" \
  --version-name "$VERSION_NAME" \
  --auto-add-overlay \
  build/res.zip

echo "==> javac"
find app/src/main/java -name '*.java' > build/sources.txt
javac --release 11 -nowarn -classpath "$PLATFORM" -d build/classes @build/sources.txt

echo "==> d8"
find build/classes -name '*.class' > build/classes.txt
"$BT/d8" "-JXmx${HEAP}" --release --lib "$PLATFORM" --min-api 24 --output build/dex @build/classes.txt
(cd build/dex && zip -q ../unsigned.apk classes.dex)

echo "==> zipalign"
if [ -x "$BT/zipalign" ]; then
  "$BT/zipalign" -f -p 4 build/unsigned.apk build/aligned.apk
else
  cp build/unsigned.apk build/aligned.apk
fi

echo "==> signing"
KEYSTORE_DIR=".keystore"
KEYSTORE="$KEYSTORE_DIR/vajra-release.jks"
PROPS="$KEYSTORE_DIR/keystore.properties"
ALIAS="vajra"
if [ ! -f "$KEYSTORE" ]; then
  mkdir -p "$KEYSTORE_DIR"
  PASS="$(head -c 24 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 20)"
  keytool -genkeypair -keystore "$KEYSTORE" -alias "$ALIAS" \
    -keyalg RSA -keysize 2048 -validity 10000 \
    -storepass "$PASS" -keypass "$PASS" \
    -dname "CN=Vajra Image Guard, OU=Open Source, O=Vajra, C=IN" >/dev/null 2>&1
  printf 'storePassword=%s\nkeyPassword=%s\nalias=%s\n' "$PASS" "$PASS" "$ALIAS" > "$PROPS"
  chmod 600 "$PROPS" "$KEYSTORE"
  echo "    generated a new signing key in $KEYSTORE_DIR (keep it to ship updates)"
fi
PASS="$(sed -n 's/^storePassword=//p' "$PROPS")"

"$BT/apksigner" "-JXmx${HEAP}" sign \
  --ks "$KEYSTORE" --ks-pass "pass:$PASS" --key-pass "pass:$PASS" \
  --min-sdk-version 24 \
  --out "build/$OUT" build/aligned.apk

"$BT/apksigner" "-JXmx${HEAP}" verify --print-certs "build/$OUT"

mkdir -p ../docs/download
cp "build/$OUT" "../docs/download/$OUT"
echo
echo "APK:  android/build/$OUT  ($(du -h "build/$OUT" | cut -f1))"
echo "Also copied to docs/download/$OUT for the website."
