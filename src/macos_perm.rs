//! macOS-only: explicit microphone permission request via AVFoundation.
//!
//! cpal opens the audio unit directly and on Apple Silicon that path does
//! NOT trigger the consent prompt, so without an explicit
//! `AVCaptureDevice.requestAccess(for: .audio)` call the OS just hands us
//! zeroed buffers and there is no way for the user to authorize the app.
//!
//! This module wraps the AppKit call. It blocks the calling thread until
//! the user closes the prompt (or, when permission has already been
//! decided, returns immediately).
//!
//! Build-time: depends on `objc2`, `objc2-foundation`, `objc2-av-foundation`
//! and `block2`. Compiled only on `target_os = "macos"`.

use std::sync::mpsc;
use std::time::Duration;

use block2::StackBlock;
use objc2::runtime::Bool;
use objc2_av_foundation::{AVAuthorizationStatus, AVCaptureDevice, AVMediaTypeAudio};
use objc2_foundation::NSString;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MicAuth {
    Authorized,
    Denied,
    Restricted,
    NotDetermined,
}

fn audio_media_type() -> &'static NSString {
    unsafe { AVMediaTypeAudio }.expect("AVMediaTypeAudio symbol resolved")
}

fn status_now() -> MicAuth {
    let s = unsafe { AVCaptureDevice::authorizationStatusForMediaType(audio_media_type()) };
    match s {
        AVAuthorizationStatus::Authorized => MicAuth::Authorized,
        AVAuthorizationStatus::Denied => MicAuth::Denied,
        AVAuthorizationStatus::Restricted => MicAuth::Restricted,
        _ => MicAuth::NotDetermined,
    }
}

/// Request microphone access. If the user has not yet decided, this shows
/// the standard macOS consent prompt and blocks for up to `timeout`. If
/// access is already granted or denied, returns immediately.
pub fn ensure_microphone_access(timeout: Duration) -> MicAuth {
    match status_now() {
        MicAuth::Authorized => return MicAuth::Authorized,
        MicAuth::Denied | MicAuth::Restricted => return status_now(),
        MicAuth::NotDetermined => {}
    }

    let (tx, rx) = mpsc::channel::<bool>();
    // The completion handler is called by AVFoundation on a private dispatch
    // queue. StackBlock closure uses `Bool` (objc2 wrapper) for the BOOL arg.
    let block = StackBlock::new(move |granted: Bool| {
        let _ = tx.send(granted.as_bool());
    });

    unsafe {
        AVCaptureDevice::requestAccessForMediaType_completionHandler(audio_media_type(), &block);
    }

    match rx.recv_timeout(timeout) {
        Ok(true) => MicAuth::Authorized,
        Ok(false) => MicAuth::Denied,
        Err(_) => status_now(),
    }
}
