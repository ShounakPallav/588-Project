function [xdot,alpha,n,L,D,M,T,rho] = uav_longitudinal_dyn(x,de,dt,p)
% States: x = [V; gamma; q; theta; h]
V   = x(1);
gam = x(2);
q   = x(3);
th  = x(4);
h   = x(5);

Veps = max(V,1e-3);

alpha = th - gam;

rho  = p.rho0*exp(-h/p.H);
qbar = 0.5*rho*V^2;

CL = p.CL0 + p.CLa*alpha + p.CLq*(p.c/(2*Veps))*q + p.CLde*de;
CD = p.CD0 + p.k*CL^2;
CM = p.CM0 + p.CMa*alpha + p.CMq*(p.c/(2*Veps))*q + p.CMde*de;

L = qbar*p.S*CL;
D = qbar*p.S*CD;
M = qbar*p.S*p.c*CM;

T = p.Tmax*dt;

Vdot   = (T*cos(alpha) - D)/p.m - p.g*sin(gam);
gamdot = (T*sin(alpha) + L)/(p.m*Veps) - p.g*cos(gam)/Veps;
qdot   = M/p.Iy;
thdot  = q;
hdot   = V*sin(gam);

xdot = [Vdot; gamdot; qdot; thdot; hdot];

n = L/(p.m*p.g); % load factor approx
end
