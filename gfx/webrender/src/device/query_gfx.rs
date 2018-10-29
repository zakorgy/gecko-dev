/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#[cfg(feature = "debug_renderer")]
use std::mem;

use device::FrameId;


pub trait NamedTag {
    fn get_label(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct GpuTimer<T> {
    pub tag: T,
    pub time_ns: u64,
}

#[derive(Debug, Clone)]
pub struct GpuSampler<T> {
    pub tag: T,
    pub count: u64,
}

pub struct QuerySet<T> {
    set: Vec<u32>,
    data: Vec<T>,
    pending: u32,
}

impl<T> QuerySet<T> {
    fn new() -> Self {
        QuerySet {
            set: Vec::new(),
            data: Vec::new(),
            pending: 0,
        }
    }

    fn reset(&mut self) {
        self.data.clear();
        self.pending = 0;
    }

    fn add(&mut self, value: T) -> Option<u32> {
        assert_eq!(self.pending, 0);
        self.set.get(self.data.len()).cloned().map(|query_id| {
            self.data.push(value);
            self.pending = query_id;
            query_id
        })
    }

    #[cfg(feature = "debug_renderer")]
    fn take<F: Fn(&mut T, u32)>(&mut self, fun: F) -> Vec<T> {
        let mut data = mem::replace(&mut self.data, Vec::new());
        for (value, &query) in data.iter_mut().zip(self.set.iter()) {
            fun(value, query)
        }
        data
    }
}

pub struct GpuFrameProfile<T> {
    timers: QuerySet<GpuTimer<T>>,
    samplers: QuerySet<GpuSampler<T>>,
    frame_id: FrameId,
    inside_frame: bool,
}

impl<T> GpuFrameProfile<T> {
    fn new() -> Self {
        GpuFrameProfile {
            timers: QuerySet::new(),
            samplers: QuerySet::new(),
            frame_id: FrameId::new(0),
            inside_frame: false,
        }
    }

    fn enable_timers(&mut self, _count: i32) {
        self.timers.set = Vec::new();
    }

    fn disable_timers(&mut self) {
        if !self.timers.set.is_empty() {
            self.timers.set.clear();
        }
        self.timers.set = Vec::new();
    }

    fn enable_samplers(&mut self, _count: i32) {
        self.samplers.set = Vec::new();
    }

    fn disable_samplers(&mut self) {
        if !self.samplers.set.is_empty() {
            self.samplers.set.clear();
        }
        self.samplers.set = Vec::new();
    }

    fn begin_frame(&mut self, frame_id: FrameId) {
        self.frame_id = frame_id;
        self.timers.reset();
        self.samplers.reset();
        self.inside_frame = true;
    }

    fn end_frame(&mut self) {
        self.finish_timer();
        self.finish_sampler();
        self.inside_frame = false;
    }

    fn finish_timer(&mut self) {
        debug_assert!(self.inside_frame);
        if self.timers.pending != 0 {
            self.timers.pending = 0;
        }
    }

    fn finish_sampler(&mut self) {
        debug_assert!(self.inside_frame);
        if self.samplers.pending != 0 {
            self.samplers.pending = 0;
        }
    }
}

impl<T: NamedTag> GpuFrameProfile<T> {
    fn start_timer(&mut self, tag: T) -> GpuTimeQuery {
        self.finish_timer();

        let marker = GpuMarker::new(tag.get_label());

        if let Some(_query) = self.timers.add(GpuTimer { tag, time_ns: 0 }) {
        }

        GpuTimeQuery(marker)
    }

    fn start_sampler(&mut self, tag: T) -> GpuSampleQuery {
        self.finish_sampler();

        if let Some(_query) = self.samplers.add(GpuSampler { tag, count: 0 }) {
        }

        GpuSampleQuery
    }

    #[cfg(feature = "debug_renderer")]
    fn build_samples(&mut self) -> (FrameId, Vec<GpuTimer<T>>, Vec<GpuSampler<T>>) {
        debug_assert!(!self.inside_frame);

        (
            self.frame_id,
            self.timers.take(|timer, _query| {
                timer.time_ns = 0
            }),
            self.samplers.take(|sampler, _query| {
                sampler.count = 0
            }),
        )
    }
}

impl<T> Drop for GpuFrameProfile<T> {
    fn drop(&mut self) {
        self.disable_timers();
        self.disable_samplers();
    }
}

pub struct GpuProfiler<T> {
    frames: Vec<GpuFrameProfile<T>>,
    next_frame: usize,
}

impl<T> GpuProfiler<T> {
    pub fn new() -> Self {
        const MAX_PROFILE_FRAMES: usize = 4;
        let frames = (0 .. MAX_PROFILE_FRAMES)
            .map(|_| GpuFrameProfile::new())
            .collect();

        GpuProfiler {
            next_frame: 0,
            frames,
        }
    }

    pub fn enable_timers(&mut self) {
        const MAX_TIMERS_PER_FRAME: i32 = 256;

        for frame in &mut self.frames {
            frame.enable_timers(MAX_TIMERS_PER_FRAME);
        }
    }

    pub fn disable_timers(&mut self) {
        for frame in &mut self.frames {
            frame.disable_timers();
        }
    }

    pub fn enable_samplers(&mut self) {
        const MAX_SAMPLERS_PER_FRAME: i32 = 16;
        if cfg!(target_os = "macos") {
            warn!("Expect OSX driver bugs related to sample queries")
        }

        for frame in &mut self.frames {
            frame.enable_samplers(MAX_SAMPLERS_PER_FRAME);
        }
    }

    pub fn disable_samplers(&mut self) {
        for frame in &mut self.frames {
            frame.disable_samplers();
        }
    }
}

impl<T: NamedTag> GpuProfiler<T> {
    #[cfg(feature = "debug_renderer")]
    pub fn build_samples(&mut self) -> (FrameId, Vec<GpuTimer<T>>, Vec<GpuSampler<T>>) {
        self.frames[self.next_frame].build_samples()
    }

    pub fn begin_frame(&mut self, frame_id: FrameId) {
        self.frames[self.next_frame].begin_frame(frame_id);
    }

    pub fn end_frame(&mut self) {
        self.frames[self.next_frame].end_frame();
        self.next_frame = (self.next_frame + 1) % self.frames.len();
    }

    pub fn start_timer(&mut self, tag: T) -> GpuTimeQuery {
        self.frames[self.next_frame].start_timer(tag)
    }

    pub fn start_sampler(&mut self, tag: T) -> GpuSampleQuery {
        self.frames[self.next_frame].start_sampler(tag)
    }

    pub fn finish_sampler(&mut self, _sampler: GpuSampleQuery) {
        self.frames[self.next_frame].finish_sampler()
    }

    pub fn start_marker(&mut self, label: &str) -> GpuMarker {
        GpuMarker::new( label)
    }

    pub fn place_marker(&mut self, label: &str) {
        GpuMarker::fire( label)
    }
}

#[must_use]
pub struct GpuMarker;

impl GpuMarker {
    fn new(_message: &str) -> Self {
        GpuMarker { }
    }

    fn fire(_message: &str) {
    }
}

#[must_use]
pub struct GpuTimeQuery(GpuMarker);
#[must_use]
pub struct GpuSampleQuery;
