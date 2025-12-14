%% uav_params_init.m
% Simple longitudinal UAV parameters + initial conditions ("fake trim")

clear;
close all;
% --- Vehicle / environment ---
p.m    = 13.5;      % kg
p.S    = 0.55;      % m^2
p.c    = 0.30;      % m
p.Iy   = 1.2;       % kg*m^2
p.Tmax = 50;        % N  (max thrust)
p.g    = 9.81;      % m/s^2

p.rho0 = 1.225;     % kg/m^3 (sea level)
p.H    = 8500;      % m (scale height)

% --- Aero coefficients (simple but usable) ---
p.CL0  = 0.30;
p.CLa  = 5.50;      % per rad
p.CLq  = 7.50;
p.CLde = 0.35;      % per rad

p.CD0  = 0.03;
p.k    = 0.06;      % induced drag factor

p.CM0  = 0.02;
p.CMa  = -1.00;
p.CMq  = -12.5;
p.CMde = -1.10;     % per rad (NEGATIVE means +de -> nose-down moment)

% --- Actuator limits ---
de_max_deg = 25;
de_min_deg = -25;
de_max = deg2rad(de_max_deg);
de_min = deg2rad(de_min_deg);

% throttle 0..1
dt_min = 0;
dt_max = 1;

% --- Initial conditions ("trim-ish") ---
h0   = 100;         % m
V0   = 20;          % m/s
gam0 = 0;           % rad
q0   = 0;           % rad/s

% compute alpha0 such that L ~= mg at V0,h0 (level-ish)
rho  = p.rho0*exp(-h0/p.H);
qbar = 0.5*rho*V0^2;
CLreq = (p.m*p.g)/(qbar*p.S);
alpha0 = (CLreq - p.CL0)/p.CLa;

th0 = alpha0 + gam0;   % theta = alpha + gamma

% --- Command profile settings ---
dh = 10;            % m altitude step size
h_step_time = 1;    % s

Vcmd0 = V0;         % hold airspeed constant initially

% --- Initial controller gains (just to start; you will tune later) ---
Kp_h     = 0.04;
Ki_h     = 0.01;

Kp_th    = 2.0;
Ki_th    = 0.6;
Kd_th    = 0.25;
N_th     = 20;

Kp_V     = 0.3;
Ki_V     = 0.08;

Kaw      = 5.0;
Kff_gam  = 0.0;

% --- Simulation settings ---
tstop = 30;         % seconds
