use eframe::egui;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::audio::input_device_names;
use crate::mt::Lang;
use crate::output::output_device_names;
use crate::pitch::Gender;
use crate::tts::{voice_display_name, voices_for_gender};
use crate::state::{
    SharedState, GENDER_MODE_AUTO, GENDER_MODE_FEMALE, GENDER_MODE_MALE,
};

pub struct PicoApp {
    pub state: Arc<Mutex<SharedState>>,
    debug_open: bool,
    /// "Clear" button sets this on click; the next frame consumes it.
    /// We can't clear inline because the SharedState lock is held read-only
    /// during render — flipping a flag and applying it at the top of the
    /// next update keeps the borrow rules clean.
    clear_pending: bool,
    /// Smoothed peak per lane for the VU meter (instant attack, exponential
    /// release at ~0.85 per frame ≈ 200ms half-life at 100ms repaint).
    vu_levels: [f32; Lang::ALL.len()],
}

/// Hand-painted level meter — sharp rectangles so no rounded "dot"
/// remains when the level is at or near zero. `level` is clamped to 0..1.
fn vu_bar(ui: &mut egui::Ui, level: f32, fill: egui::Color32, width: f32, height: f32, text: &str) {
    let level = level.clamp(0.0, 1.0);
    let (rect, _resp) =
        ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter();
    // Track background
    painter.rect_filled(rect, 0.0, egui::Color32::from_gray(35));
    // Filled portion (only painted when there is something to draw)
    if level * width > 0.5 {
        let fill_rect = egui::Rect::from_min_size(
            rect.left_top(),
            egui::vec2(rect.width() * level, rect.height()),
        );
        painter.rect_filled(fill_rect, 0.0, fill);
    }
    // 1px subtle border
    painter.rect_stroke(
        rect,
        0.0,
        egui::Stroke::new(1.0, egui::Color32::from_gray(60)),
    );
    if !text.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            text,
            egui::FontId::monospace(11.0),
            egui::Color32::from_gray(220),
        );
    }
}

fn channel_pair_label(start: u16, total_channels: u16) -> String {
    let l = start + 1;
    if start + 1 < total_channels {
        format!("Ch {}-{}", l, l + 1)
    } else {
        format!("Ch {} (out of range)", l)
    }
}

impl PicoApp {
    pub fn new(state: Arc<Mutex<SharedState>>) -> Self {
        Self {
            state,
            debug_open: false,
            clear_pending: false,
            vu_levels: [0.0; Lang::ALL.len()],
        }
    }
}

impl eframe::App for PicoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint_after(Duration::from_millis(100));

        if self.clear_pending {
            self.state.lock().unwrap().clear_all();
            self.clear_pending = false;
        }

        let s = self.state.lock().unwrap();
        let active = s.controls.active.load(Ordering::Relaxed);
        let source = s.controls.get_source();

        // PERMISSION BANNER — only when we detected silence at startup.
        if let Some(msg) = &s.mic_permission_warning {
            egui::TopBottomPanel::top("perm-banner").show(ctx, |ui| {
                let banner_color = egui::Color32::from_rgb(180, 50, 50);
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.colored_label(banner_color, egui::RichText::new("⚠ MICROPHONE BLOCKED").strong());
                    ui.label(
                        egui::RichText::new(msg)
                            .small()
                            .color(egui::Color32::from_gray(200)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Open Privacy settings").clicked() {
                            #[cfg(target_os = "macos")]
                            {
                                let _ = std::process::Command::new("open")
                                    .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")
                                    .spawn();
                            }
                        }
                    });
                });
                ui.add_space(4.0);
            });
        }

        // TOP STATUS BAR
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let (dot_color, label) = if s.listening && active {
                    (egui::Color32::from_rgb(50, 200, 80), "LIVE")
                } else if s.listening && !active {
                    (egui::Color32::from_rgb(240, 180, 60), "STOPPED")
                } else {
                    (egui::Color32::from_rgb(120, 120, 120), "IDLE")
                };
                ui.colored_label(dot_color, egui::RichText::new("●").size(18.0));
                ui.label(egui::RichText::new(label).strong());
                ui.separator();

                let btn_label = if active { "⏸  Stop" } else { "▶  Start" };
                if ui.button(btn_label).clicked() {
                    s.controls.active.store(!active, Ordering::Relaxed);
                }

                ui.separator();
                ui.label("source:");
                let mut src_idx = source.index();
                egui::ComboBox::from_id_source("source-lang")
                    .selected_text(source.code().to_uppercase())
                    .show_ui(ui, |ui| {
                        for (i, lang) in Lang::ALL.iter().enumerate() {
                            if ui
                                .selectable_value(&mut src_idx, i, lang.code().to_uppercase())
                                .clicked()
                            {
                                s.controls.source.store(i as u8, Ordering::Relaxed);
                                // Bump the session token so the worker
                                // throws away the gender tracker built up
                                // for the previous speaker/language.
                                s.controls.session_token.fetch_add(1, Ordering::Relaxed);
                                // Reset target enables: previous source is
                                // now translatable again, new source isn't.
                                for (j, l) in Lang::ALL.iter().enumerate() {
                                    let on = j != i;
                                    s.controls.target_enabled[j].store(on, Ordering::Relaxed);
                                    s.controls.set_voice_override(*l, None);
                                }
                            }
                        }
                    });

                ui.separator();
                ui.label("voice:");
                let cur_mode = s.controls.gender_mode.load(Ordering::Relaxed);
                for (mode, label) in [
                    (GENDER_MODE_AUTO, "Auto"),
                    (GENDER_MODE_MALE, "♂ Male"),
                    (GENDER_MODE_FEMALE, "♀ Female"),
                ] {
                    if ui
                        .selectable_label(cur_mode == mode, label)
                        .clicked()
                    {
                        s.controls.gender_mode.store(mode, Ordering::Relaxed);
                        // Switching gender mode invalidates any per-lane
                        // voice override the user picked under the old
                        // mode (e.g. selected `am_michael` while in Male,
                        // then flipped to Female — the override is no
                        // longer in the list). Clear them all so the
                        // pipeline falls back to gender-driven defaults.
                        for l in Lang::ALL {
                            s.controls.set_voice_override(*l, None);
                        }
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("{} chunks", s.chunks_processed));
                    ui.separator();
                    ui.label(format!("mt: {}", s.ollama_model));
                });
            });
            ui.add_space(4.0);
        });

        // DEVICES ROW
        egui::TopBottomPanel::top("devices").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("DEVICES")
                        .small()
                        .color(egui::Color32::from_gray(160)),
                );
                ui.separator();

                // INPUT
                ui.label("in:");
                let inputs = s.controls.input_devices.lock().unwrap().clone();
                let mut current_in = s.input_device.clone();
                egui::ComboBox::from_id_source("input-device")
                    .selected_text(if current_in.is_empty() { "<auto>" } else { &current_in })
                    .width(260.0)
                    .show_ui(ui, |ui| {
                        if ui.selectable_value(&mut current_in, "<auto>".to_string(), "<auto>").clicked() {
                            *s.controls.requested_input.lock().unwrap() = None;
                            s.controls.restart_flag.store(true, Ordering::Relaxed);
                        }
                        for d in &inputs {
                            if ui.selectable_value(&mut current_in, d.clone(), d).clicked() {
                                *s.controls.requested_input.lock().unwrap() = Some(d.clone());
                                s.controls.restart_flag.store(true, Ordering::Relaxed);
                            }
                        }
                    });

                ui.separator();

                // OUTPUT
                ui.label("out:");
                let outputs = s.controls.output_devices.lock().unwrap().clone();
                let mut current_out = s.output_device.clone();
                egui::ComboBox::from_id_source("output-device")
                    .selected_text(if current_out.is_empty() { "<auto>" } else { &current_out })
                    .width(260.0)
                    .show_ui(ui, |ui| {
                        if ui.selectable_value(&mut current_out, "<auto>".to_string(), "<auto>").clicked() {
                            *s.controls.requested_output.lock().unwrap() = None;
                            s.controls.restart_flag.store(true, Ordering::Relaxed);
                        }
                        for d in &outputs {
                            if ui.selectable_value(&mut current_out, d.clone(), d).clicked() {
                                *s.controls.requested_output.lock().unwrap() = Some(d.clone());
                                s.controls.restart_flag.store(true, Ordering::Relaxed);
                            }
                        }
                    });

                ui.label(format!("({} ch)", s.output_channels));

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .button("↻")
                        .on_hover_text("Rescan input/output devices (e.g. after plugging headphones)")
                        .clicked()
                    {
                        // cpal device enumeration is cheap (a few ms) and
                        // safe to run from the UI thread.
                        *s.controls.input_devices.lock().unwrap() = input_device_names();
                        *s.controls.output_devices.lock().unwrap() = output_device_names();
                    }
                });
            });

            // Session-save row: where the per-call WAV recording lives.
            if let Some(dir) = s.session_dir.clone() {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("REC")
                            .small()
                            .color(egui::Color32::from_rgb(220, 80, 80)),
                    );
                    let display = dir.display().to_string();
                    ui.label(
                        egui::RichText::new(&display)
                            .small()
                            .color(egui::Color32::from_gray(180)),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Open folder").clicked() {
                            #[cfg(target_os = "macos")]
                            {
                                let _ = std::process::Command::new("open").arg(&dir).spawn();
                            }
                            #[cfg(not(target_os = "macos"))]
                            {
                                let _ = std::process::Command::new("xdg-open").arg(&dir).spawn();
                            }
                        }
                    });
                });
            }
            ui.add_space(4.0);
        });

        // MIXER ROW — one strip per lang: enable / channel-pair / volume fader
        egui::TopBottomPanel::top("mixer").show(ctx, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("MIXER")
                        .small()
                        .color(egui::Color32::from_gray(160)),
                );
            });
            let channels = s.output_channels as u16;
            ui.columns(Lang::ALL.len(), |cols| {
                for (i, lang) in Lang::ALL.iter().enumerate() {
                    let ui = &mut cols[i];
                    let is_source = *lang == source;

                    // Header: lang code
                    ui.horizontal(|ui| {
                        let color = if is_source {
                            egui::Color32::from_rgb(80, 140, 220)
                        } else {
                            egui::Color32::from_gray(220)
                        };
                        ui.colored_label(
                            color,
                            egui::RichText::new(lang.code().to_uppercase())
                                .size(16.0)
                                .strong(),
                        );
                        if is_source {
                            ui.label(
                                egui::RichText::new("SRC")
                                    .small()
                                    .color(egui::Color32::from_rgb(80, 140, 220)),
                            );
                        }
                    });

                    // Enable checkbox
                    let en_atomic = &s.controls.target_enabled[i];
                    let mut checked = en_atomic.load(Ordering::Relaxed);
                    ui.add_enabled_ui(!is_source, |ui| {
                        if ui.checkbox(&mut checked, "enabled").changed() {
                            en_atomic.store(checked, Ordering::Relaxed);
                        }
                    });

                    // Channel-pair dropdown
                    let cur = s.controls.channel_starts[i].load(Ordering::Relaxed);
                    let pair_label = channel_pair_label(cur, channels);
                    egui::ComboBox::from_id_source(format!("ch-{}", i))
                        .selected_text(pair_label)
                        .width(110.0)
                        .show_ui(ui, |ui| {
                            for start in 0..channels.saturating_sub(1) {
                                let label = channel_pair_label(start, channels);
                                if ui
                                    .selectable_label(start == cur, label)
                                    .clicked()
                                {
                                    s.controls.channel_starts[i].store(start, Ordering::Relaxed);
                                }
                            }
                        });

                    // Volume fader
                    let mut vol = s.controls.volume(*lang);
                    ui.add(
                        egui::Slider::new(&mut vol, 0.0..=1.5)
                            .text("gain")
                            .custom_formatter(|v, _| {
                                if v <= 0.001 {
                                    "-∞ dB".to_string()
                                } else {
                                    format!("{:+.1} dB", 20.0 * v.log10())
                                }
                            }),
                    );
                    s.controls.set_volume(*lang, vol);

                    // VU meter — peak from the audio callback, smoothed in
                    // self.vu_levels (instant attack, exponential release).
                    // Shows what's leaving the device for this lane.
                    let raw_peak = s.controls.output_peak(*lang).clamp(0.0, 1.0);
                    let smoothed = &mut self.vu_levels[i];
                    if raw_peak > *smoothed {
                        *smoothed = raw_peak;
                    } else {
                        *smoothed = (*smoothed * 0.85).max(raw_peak);
                    }
                    let level = *smoothed;
                    let bar_color = if level >= 0.85 {
                        egui::Color32::from_rgb(220, 60, 60)
                    } else if level >= 0.5 {
                        egui::Color32::from_rgb(240, 180, 60)
                    } else {
                        egui::Color32::from_rgb(50, 200, 80)
                    };
                    let vu_text = if level > 0.001 {
                        format!("{:+.0} dB", 20.0 * level.log10())
                    } else {
                        "-∞ dB".to_string()
                    };
                    vu_bar(ui, level, bar_color, 140.0, 16.0, &vu_text);

                    // Voice picker. Visible only when the user has fixed
                    // the gender (Male / Female) — in Auto mode the pitch
                    // tracker picks the voice per utterance and any
                    // override would defeat the purpose.
                    let mode = s.controls.gender_mode.load(Ordering::Relaxed);
                    let forced_gender = match mode {
                        GENDER_MODE_MALE => Some(Gender::Male),
                        GENDER_MODE_FEMALE => Some(Gender::Female),
                        _ => None,
                    };
                    if let Some(g) = forced_gender {
                        let voices = voices_for_gender(*lang, g);
                        if !voices.is_empty() {
                            let current_override = s.controls.voice_override(*lang);
                            let display_current = current_override
                                .as_deref()
                                .map(voice_display_name)
                                .unwrap_or_else(|| "default".to_string());
                            egui::ComboBox::from_id_source(format!("voice-{}", i))
                                .selected_text(display_current)
                                .width(140.0)
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_label(current_override.is_none(), "default")
                                        .clicked()
                                    {
                                        s.controls.set_voice_override(*lang, None);
                                    }
                                    for &v in voices {
                                        let selected = current_override.as_deref() == Some(v);
                                        if ui
                                            .selectable_label(selected, voice_display_name(v))
                                            .clicked()
                                        {
                                            s.controls.set_voice_override(*lang, Some(v.to_string()));
                                        }
                                    }
                                });
                        }
                    } else {
                        ui.label(
                            egui::RichText::new("voice: auto")
                                .small()
                                .color(egui::Color32::from_gray(140)),
                        );
                    }
                }
            });
            ui.add_space(6.0);
        });

        // BOTTOM PANEL — mic meter, VAD thresholds, debug toggle, optional log
        let mic_rms = s.controls.input_rms();
        let mut rms_high = s.controls.vad_rms_high();
        let mut rms_low = s.controls.vad_rms_low();
        let debug_log = s.debug_log.clone();
        egui::TopBottomPanel::bottom("bottom")
            .resizable(true)
            .min_height(70.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);

                // Row 1: live mic meter + VAD thresholds
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("MIC")
                            .small()
                            .color(egui::Color32::from_gray(160)),
                    );
                    let bar_max = 0.05_f32;
                    let frac = (mic_rms / bar_max).clamp(0.0, 1.0);
                    let bar_color = if mic_rms >= rms_high {
                        egui::Color32::from_rgb(50, 200, 80)
                    } else if mic_rms >= rms_low {
                        egui::Color32::from_rgb(240, 180, 60)
                    } else {
                        egui::Color32::from_rgb(120, 120, 120)
                    };
                    vu_bar(ui, frac, bar_color, 220.0, 16.0, &format!("{:.4}", mic_rms));

                    ui.separator();
                    ui.label("trig:");
                    if ui
                        .add(
                            egui::Slider::new(&mut rms_high, 0.001..=0.05)
                                .logarithmic(true)
                                .fixed_decimals(4),
                        )
                        .changed()
                    {
                        s.controls.set_vad_rms_high(rms_high);
                        // Keep low ≤ high.
                        if rms_low > rms_high {
                            rms_low = rms_high * 0.5;
                            s.controls.set_vad_rms_low(rms_low);
                        }
                    }
                    ui.label("end:");
                    if ui
                        .add(
                            egui::Slider::new(&mut rms_low, 0.0005..=0.05)
                                .logarithmic(true)
                                .fixed_decimals(4),
                        )
                        .changed()
                    {
                        s.controls.set_vad_rms_low(rms_low.min(rms_high));
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .selectable_label(self.debug_open, "🐛 debug")
                            .clicked()
                        {
                            self.debug_open = !self.debug_open;
                        }
                        if ui.button("Clear").on_hover_text("Wipe history, transcript, and debug log").clicked() {
                            self.clear_pending = true;
                        }
                    });
                });

                // Row 2: timings (always visible)
                ui.label(
                    egui::RichText::new(format!(
                        "stt {} ms · mt [en {} · fr {} · es {} · it {}] · tts [en {} · fr {} · es {} · it {}]",
                        s.stt_ms,
                        s.mt_ms_en, s.mt_ms_fr, s.mt_ms_es, s.mt_ms_it,
                        s.tts_ms_en, s.tts_ms_fr, s.tts_ms_es, s.tts_ms_it,
                    ))
                    .monospace()
                    .small(),
                );

                if self.debug_open {
                    ui.separator();
                    ui.label(
                        egui::RichText::new(format!("DEBUG ({} entries, newest first)", debug_log.len()))
                            .small()
                            .color(egui::Color32::from_gray(160)),
                    );
                    egui::ScrollArea::vertical()
                        .max_height(220.0)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            for line in debug_log.iter() {
                                ui.label(egui::RichText::new(line).monospace().small());
                            }
                        });
                }

                ui.add_space(4.0);
            });

        // CENTER
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(6.0);
            ui.heading(format!("{} (source)", source.code().to_uppercase()));
            ui.add(
                egui::Label::new(
                    egui::RichText::new(if s.src_latest.is_empty() {
                        "…waiting for speech…"
                    } else {
                        &s.src_latest
                    })
                    .size(22.0),
                )
                .wrap(),
            );

            ui.add_space(12.0);
            ui.separator();
            ui.add_space(12.0);

            let targets: Vec<Lang> = Lang::ALL
                .iter()
                .copied()
                .filter(|l| *l != source && s.controls.is_target_enabled(*l))
                .collect();

            if targets.is_empty() {
                ui.label(
                    egui::RichText::new("No targets enabled — pick one or more above.")
                        .color(egui::Color32::from_gray(160)),
                );
            } else {
                ui.columns(targets.len(), |cols| {
                    for (idx, lang) in targets.iter().enumerate() {
                        cols[idx].heading(lang.code().to_uppercase());
                        cols[idx].add(
                            egui::Label::new(
                                egui::RichText::new(if s.translation_for(*lang).is_empty() {
                                    "—"
                                } else {
                                    s.translation_for(*lang)
                                })
                                .size(18.0),
                            )
                            .wrap(),
                        );
                    }
                });
            }

            ui.add_space(12.0);
            ui.separator();
            ui.add_space(6.0);

            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.label(egui::RichText::new("HISTORY").strong().small());
                    ui.add_space(2.0);
                    for entry in s.history.iter() {
                        ui.group(|ui| {
                            ui.label(
                                egui::RichText::new(format!(
                                    "[{}] {}",
                                    entry.src_code.to_uppercase(),
                                    entry.src_text
                                ))
                                .size(13.0)
                                .strong(),
                            );
                            let tr_str = entry
                                .translations
                                .iter()
                                .map(|(l, t)| format!("{}  {}", l.code().to_uppercase(), t))
                                .collect::<Vec<_>>()
                                .join("\n");
                            ui.label(
                                egui::RichText::new(tr_str)
                                    .size(12.0)
                                    .color(egui::Color32::from_gray(160)),
                            );
                        });
                        ui.add_space(4.0);
                    }
                });
        });
    }
}
