#pragma once

struct SoccerParams {
    float acceleration;
    float relative_ball_width;
    float additional_model_delay; 
    float confidence_threshold;
    
    int max_lost_frames;

    // height and speed at which we trigger
    float height_trigger_soft;
    float fall_speed_trigger_soft;

    // speed at which we trigger regardless of height
    float fall_speed_trigger_hard;
};