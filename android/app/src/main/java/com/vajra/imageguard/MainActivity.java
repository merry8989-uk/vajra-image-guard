package com.vajra.imageguard;

import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Base64;
import android.view.ViewGroup;
import android.webkit.JavascriptInterface;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;

/**
 * The whole app: a WebView pointed at the bundled copy of the website, plus a
 * bridge that lets the page hand a finished PNG to the system media store.
 *
 * Deliberately framework-only (no AndroidX, no Gradle) so it builds with aapt2,
 * javac, d8 and apksigner, and so the APK stays under 250 KB.
 */
public class MainActivity extends Activity {

    private static final int REQ_PICK = 1001;
    private static final int BG_COLOR = 0xFF07080B;

    private WebView web;
    private ValueCallback<Uri[]> pendingPicker;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);

        web = new WebView(this);
        web.setBackgroundColor(BG_COLOR);
        setContentView(web, new ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setAllowFileAccess(true);
        s.setAllowContentAccess(true);
        s.setSupportZoom(false);
        s.setBuiltInZoomControls(false);
        s.setTextZoom(100);

        web.setWebViewClient(new WebViewClient());
        web.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback,
                                             FileChooserParams params) {
                if (pendingPicker != null) {
                    pendingPicker.onReceiveValue(null);
                }
                pendingPicker = callback;

                Intent pick = new Intent(Intent.ACTION_GET_CONTENT);
                pick.addCategory(Intent.CATEGORY_OPENABLE);
                pick.setType("image/*");
                pick.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
                try {
                    startActivityForResult(Intent.createChooser(pick, "Choose an image"), REQ_PICK);
                } catch (Exception e) {
                    pendingPicker = null;
                    return false;
                }
                return true;
            }
        });

        web.addJavascriptInterface(new Bridge(), "VajraAndroid");
        web.loadUrl("file:///android_asset/www/index.html");
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode != REQ_PICK) {
            super.onActivityResult(requestCode, resultCode, data);
            return;
        }
        if (pendingPicker == null) {
            return;
        }
        Uri[] result = null;
        if (resultCode == RESULT_OK && data != null) {
            if (data.getClipData() != null) {
                int count = data.getClipData().getItemCount();
                result = new Uri[count];
                for (int i = 0; i < count; i++) {
                    result[i] = data.getClipData().getItemAt(i).getUri();
                }
            } else if (data.getData() != null) {
                result = new Uri[] { data.getData() };
            }
        }
        pendingPicker.onReceiveValue(result);
        pendingPicker = null;
    }

    @Override
    public void onBackPressed() {
        if (web != null && web.canGoBack()) {
            web.goBack();
        } else {
            super.onBackPressed();
        }
    }

    /** Methods exposed to the page as window.VajraAndroid. */
    private class Bridge {

        private Uri lastSaved;

        @JavascriptInterface
        public boolean saveImage(String dataUrl, String fileName) {
            return write(dataUrl, fileName, false);
        }

        @JavascriptInterface
        public boolean shareImage(String dataUrl, String fileName) {
            return write(dataUrl, fileName, true);
        }

        @JavascriptInterface
        public String platform() {
            return "android-" + Build.VERSION.SDK_INT;
        }

        private boolean write(String dataUrl, String fileName, boolean shareAfter) {
            try {
                Bitmap bitmap = decode(dataUrl);
                if (bitmap == null) {
                    toast("Could not decode the image");
                    return false;
                }
                String name = sanitize(fileName);
                boolean png = name.toLowerCase().endsWith(".png");

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    ContentValues values = new ContentValues();
                    values.put(MediaStore.Images.Media.DISPLAY_NAME, name);
                    values.put(MediaStore.Images.Media.MIME_TYPE, png ? "image/png" : "image/jpeg");
                    values.put(MediaStore.Images.Media.RELATIVE_PATH,
                            Environment.DIRECTORY_PICTURES + "/Vajra");
                    values.put(MediaStore.Images.Media.IS_PENDING, 1);

                    Uri target = getContentResolver()
                            .insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
                    if (target == null) {
                        toast("Media store refused the file");
                        return false;
                    }
                    try (OutputStream out = getContentResolver().openOutputStream(target)) {
                        if (out == null) throw new IllegalStateException("no output stream");
                        bitmap.compress(png ? Bitmap.CompressFormat.PNG : Bitmap.CompressFormat.JPEG,
                                png ? 100 : 95, out);
                    }
                    values.clear();
                    values.put(MediaStore.Images.Media.IS_PENDING, 0);
                    getContentResolver().update(target, values, null, null);

                    lastSaved = target;
                    toast("Saved to Pictures/Vajra");
                    if (shareAfter) share(target, png ? "image/png" : "image/jpeg");
                    return true;
                }

                // Pre-Q: no media store scoping to fight with, so use the app's own
                // external pictures directory. No permission needed either way.
                File dir = new File(getExternalFilesDir(Environment.DIRECTORY_PICTURES), "Vajra");
                if (!dir.exists() && !dir.mkdirs()) {
                    toast("Could not create the output folder");
                    return false;
                }
                File file = new File(dir, name);
                try (FileOutputStream out = new FileOutputStream(file)) {
                    bitmap.compress(png ? Bitmap.CompressFormat.PNG : Bitmap.CompressFormat.JPEG,
                            png ? 100 : 95, out);
                }
                toast("Saved to " + file.getAbsolutePath());
                return true;
            } catch (Exception e) {
                toast("Save failed: " + e.getClass().getSimpleName());
                return false;
            }
        }

        private Bitmap decode(String dataUrl) {
            int comma = dataUrl.indexOf(',');
            String payload = comma >= 0 ? dataUrl.substring(comma + 1) : dataUrl;
            byte[] bytes = Base64.decode(payload, Base64.DEFAULT);
            return BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
        }

        private String sanitize(String name) {
            if (name == null || name.trim().isEmpty()) return "vajra.png";
            String cleaned = name.replaceAll("[^A-Za-z0-9._-]", "_");
            if (!cleaned.contains(".")) cleaned = cleaned + ".png";
            return cleaned;
        }

        private void share(Uri uri, String mime) {
            final Intent send = new Intent(Intent.ACTION_SEND);
            send.setType(mime);
            send.putExtra(Intent.EXTRA_STREAM, uri);
            send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    try {
                        startActivity(Intent.createChooser(send, "Share protected image"));
                    } catch (Exception ignored) {
                        // no share target installed; the file is already saved
                    }
                }
            });
        }

        private void toast(final String message) {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    Toast.makeText(MainActivity.this, message, Toast.LENGTH_SHORT).show();
                }
            });
        }
    }
}
