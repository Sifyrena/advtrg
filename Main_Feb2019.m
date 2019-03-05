%% Header Section
% OVEAL_ReWrite :)
% You get all convex hulls ... Soon .

clear;
clc;
clf;

% Maximum Number of Iterations (for Searching Only)
Iter = 4000;

%% Input Parameters
%  Multiples of pi/2
AngleA = 1/2;
AngleB = 1/200;
% Fun
Fagnano = false;

eps = -1e-4;
PertC = 0.0;
PertTh = 0.0;

%% Design Elements

% Edge: #3e7aa2
DesEdge = [62 122 162]./255;
DesNorm = [162	102	62]./255;
DesBG = [1 1 1]./255;
DesOrb = [0.5 0.5 0.5];

%% Initial Conditions
% c lives between 0 and 3
% Th0P lives in (0, 2) -> (0,pi) ....... 


c0 = 0.75;

% Evolve from Positive Absolute x-Direction
Th0P = 1;

%% Periodic Motion in Triangular Billiards
% FW Dynamical Trig Billiard Table. Rebuilt (SummerJoy)
% For MATLAB r2018a and Coco
%
%% Development Log
%
%  3-D orbit Plots. Dec 5th 2018.
%  Orbit Word Intepreter v1, Dec 7th 2018
%  Built-In Birkhof-Poincare Map, Dec 7th 2018, *DnF*
%  Embedded zero fn (x0,AnV0) -> (c-x0,AnV-AnV0), Dec 12th 2018
%  Numerical Period Searcher, Dec 13th 2018
%  Advanced (Partial) Modularity Definition To Use With CoCo, 22 Jan
%  --- Separate File
%  Media Workshop for Powerpoint Session, 22 Jan (Separate File)
%
%% Work In Progress Features
%
%  Complete Dynamic Analysis Suites
%  Built-in Function Definition
%  Vertex Hitting Determination
%
%% Long-Term Planned Features
%  Full Set of McBilliards Functions
%
%% End Development Header
% For Optional Functionalities, Uncomment Code Between Neighbouring '...'

%% Global Parameters
%
% StateSpace: Name of Each Edge, L := Left. R := Right. B := Base 
% StateSpaceDesig: (0,1) -> B; (1,2) -> R; (2,3) -> L
% ..................................................................

PtA = [0;0];
PtB = [1;0];

%% Hey_Anna
% Input Digestion
AnA = AngleA*pi/2;
AnB = AngleB*pi/2;
% Some more

Th0 = Th0P*pi/2;

Th0 = wrapToPi(2*Th0)/2;

%% The Third Point On Top
% Investigate the Accuracy [INSERT NUMERICAL EVALUATION HERE]

PtC = [tan(AnB)/(tan(AnA)+tan(AnB));tan(AnA)*tan(AnB)/(tan(AnA)+tan(AnB))];
PtCx = PtC(1);
PtCy = PtC(2);



%% PSpace

figure(1);
title('FW Trig Billiard Toolbox')
clf;
subplot(3,3,1)
set(gca,'Color',DesBG)
hold on;

line([0 1], [1 1],'Color','white');
line([0 0], [1 0],'Color','white');
line([0 0], [0 1],'Color','white');
line([1 1], [0 1],'Color','white');
line([0 1], [1 0],'Color',DesEdge);
line([0 1], [0 1],'Color',(DesEdge.^2));

plot(AngleA,AngleB,'o','Color','c');

xlim([0 1]);
ylim([0 1]);
axis equal;
grid on;
title('Parameter Space')
hold off;


subplot(3,3,4)
hold on;
set(gca,'Color',DesBG);
xlim([-1 1]);
ylim([-1 1]);
if Fagnano
   plot(0,0,'o','Color','cyan','LineWidth',1); 
else
    plot([0 cos(Th0)],[0 sin(Th0)],'-','Color','cyan','LineWidth',1);
end


SigAngles = [0 1/6 1/3 1/2 2/3 5/6 1 1/4 3/4 7/6 4/3 3/2 5/3 11/6 7/4 5/4 2 0.5*AngleA 0.5*AngleB];

for pl = 1:length(SigAngles)
plot([0 cos(SigAngles(pl)*pi)],[0 sin(SigAngles(pl)*pi)],'-.','Color',[0.4 0.4 0.4]);
end

viscircles([0 0],1,'Color','w','LineWidth', 0.1)
grid on;
axis equal;
title('Initial Ray Direction')
hold off;

%% Edge Cushions Defined on the Interval (0,1)
% OLD
EdgeL = @(x) tan(AnA)*x;
EdgeR = @(x) tan(AnB)*(1-x);
EdgeB = @(x) 0*x;

% New Symbolic Processing Engine
% syms Edges(t)
% Edges(t) = piecewise(floor(t)==0, [t;0], floor(t)==1, [1-(t-1)*(1-PtCx);PtCy*(t-1)],floor(t)==2,[PtCx*(3-t);PtCy*(3-t)],[0;0]);

%% Perimeter Determination
% Centroid
PtCe = (PtA+PtB+PtC)./3;
% Vector BC
PtBC = PtC - [1;0];
% The Lengths of Three Segments of Boundary
dL = sqrt(PtC.'*PtC);
dB = 1;
dR = sqrt(PtBC.'*PtBC);

%% Pre-Run Dialog Box
fprintf('FW Triangle Plotter Rewrite 2019 \n')
fprintf('OVEAL est. 2017 \n')
fprintf('Performance Upgrade Underway. \nPlease Read Header for Implemented Functionalities. \n')
fprintf('\n=================================\n')
fprintf(['Maximum Number of Iterations This Time is ', num2str(Iter)])
fprintf('\n=================================')
fprintf(['\nThe Two Base Angles Are \nA: ',num2str(AngleA),' and \nB: ' ,num2str(AngleB),' times of quarter tau'])
fprintf('\n---------------------------------')
fprintf(['\nThe Third Angle is \nC: ',num2str(2-AngleA-AngleB), ' times of quarter tau'])

%% Orbit Plotter
%
subplot(3,3,[2 3 5 6]);
hold on;
grid on;
axis equal; % Innocent People Will Die Without This Line
set(gca,'Color',DesBG)
fplot(EdgeB,[0,1],'Color',DesEdge);
fplot(EdgeL,[0,PtCx],'Color',DesEdge);
fplot(EdgeR,[PtCx,1],'Color',DesEdge);

N0 = [0 1];
N1 = -[cos(pi/2-AnB) sin(pi/2-AnB)];
N2 = -[cos(pi/2+AnA) sin(pi/2+AnA)];

Or0 = [0.5 0];
Or1 = [0.5 * (PtCx+1),0.5*PtCy];
Or2 = [0.5 * PtCx,0.5*PtCy];


%% [Opt] Fagnano's Problem
% Fugnano Problem Finds The Inscribed Triangle of Minimum Perimeter. 
% This has proven to be the orthic triangle, and it satisfies the billiard properties
% (Reflection Law).
% 
% Orthic triangles degenerate into a line segment for right triangles and 
% do not exist for obtuse triangles.


% ==
if Fagnano
    
    if 2-AngleA-AngleB > 1
        fprintf('\nThe Triangle is Right or Obtuse. Continuing');

    else
        fprintf('\nUsing Fagnano Triangle Orbit, Type 123123. ')
        c0 = PtCx;
        HeightL = @(x) cot(AnA)*(1-x);
        HeightR = @(x) cot(AnB)*(x);
        NextPoint = @(x) cot(AnA)*(1-x) - tan(AnA)*x;
        NextPoint2 = @(x) cot(AnB)*(x) - tan(AnB)*(1-x);
        nextpoint = fzero(NextPoint,0);
        nextpoint2 = fzero(NextPoint2,0);
    
        fplot(HeightL,[nextpoint,1])
        fplot(HeightR,[0,nextpoint2])
        Th0 = - atan(tan(AnA)*nextpoint/(c0-nextpoint));
        Th0P = Th0/pi;
        
        a_cF = c0;
        a_ThFP = Th0P;
    end
end

c0 = c0+ PertC;
Th0 = Th0 + PertTh;
fprintf('\n=================================\n')
fprintf(['The Initial Laser Beam Enters at \n', num2str(c0), ' Normalized Units along the perimeter, and \n', num2str(Th0P), ' times pi/2 rad from positive x direction \n']);

% ==

%% Locating Initial Points FOR PLOTTING

if floor(c0) == 0
    x0 = dB*c0;
    y0 = 0;
    
elseif floor(c0) == 1
    x0 = PtCx + (2-c0)*(1-PtCx);
    y0 = EdgeR(x0);
    
elseif floor(c0) == 2
    x0 = PtCx*(3-c0);
    y0 = EdgeL(x0);
    
end
plot(x0,y0,'ko');


%% Old Linear Operators
% Temporary Idea: What if we differentiate \del \Delta every time?

% Temporary Idea2 * WORK ON THIS*
% State Variables (c,Th) and Plotter Variables ARE
% DIFFERENT. Consider New Ways to Influence the angles! WE DO NOT NEED
% O(2) MATRICES.

%% New Maps

% Map from R: th -> pi - 2AnB - th
% Map from L: th -> pi + 2AnA - th
% Map from B: th -> - th

% Getting everyone back in range: x -> atan(tan(x))


%% Runtime Variables Init

% Debug Features 

cLog = zeros([Iter+1,1]);
ThLog = zeros([Iter+1,1]);
PtLog = zeros([2,Iter+1]);
WordLog = zeros([Iter+1,1]);

LogicDebug = zeros([3,Iter]);
% Regular Variables
c = c0;
Th = Th0;
Pt0 = [x0; y0];

% Debug Features Cont'd
WordLog(1) = floor(c0);
cLog(1) = c0;
ThLog(1) = Th0;
PtLog(:,1) = Pt0;

%% Starting Run

xR = x0;
yR = y0;
    
for n = 1:Iter

% New Zerofindr %

    ZeroL = @(x) tan(AnA)*x + tan(Th)*(xR-x) - yR;
    ZeroR = @(x) tan(AnB)*(1-x) + tan(Th)*(xR-x) - yR;
    ZeroB = @(x) tan(Th)*(x-xR)+yR;
    



    % New Parallel Detection

if Th == 0
    zeroB = NaN;
    zeroL = fzero(ZeroL,0);
    zeroR = fzero(ZeroR,0);
    
elseif Th == AnA
    zeroL = NaN;
    zeroB = fzero(ZeroB,0);
	zeroR = fzero(ZeroR,0);
    
elseif Th == pi - AnB
    zeroR = NaN;
    zeroL = fzero(ZeroL,0);
    zeroB = fzero(ZeroB,0);
else
	zeroL = fzero(ZeroL,0);
    zeroB = fzero(ZeroB,0);
    zeroR = fzero(ZeroR,0);
    
end


if floor(c) == 0 % B
    zeroB = NaN;
    if 0<=zeroL&&zeroL<=PtCx
%        disp('L');
            % -> L
            c = 3-zeroL/PtCx;
            Th = 2*AnA - Th;
            
            
    elseif PtCx<=zeroR&&zeroR<=1
%         disp('R')
            % -> R
            c = 2 - (zeroR-PtCx)/(1-PtCx);
            Th = - 2*AnB - Th;
    end

elseif floor(c) == 1 % R
    zeroR = NaN;
    if 0<=zeroL&&zeroL<=PtCx
%         disp('L')
            % -> L
            c = 3-zeroL/PtCx;
            Th = 2*AnA - Th;
            
    elseif 0<=zeroB&&zeroB<=1
%         disp('B')
            % -> B
            c = zeroB;
            Th = - Th;
    end
    
elseif floor(c) == 2 % L
    zeroL = NaN;
    
    if PtCx<=zeroR&&zeroR<=1
%         disp('R')
            % -> R
            c = 2 - (zeroR-PtCx)/(1-PtCx);
            Th = - 2*AnB - Th;
            
    elseif 0<=zeroB&&zeroB<=1
%          disp('B')
            % -> B
            c = zeroB;
            Th = - Th;
    end
    


end % Zero Logic Concludes

Th = atan(tan(Th));

if floor(c) == 0
    xR = c;
    yR = 0;
    
elseif floor(c) == 1
    xR = PtCx + (2-c)*(1-PtCx);
    yR = EdgeR(xR);
    
elseif floor(c) == 2
    xR = PtCx*(3-c);
    yR = EdgeL(xR);
    
end % Plotter Locator

% Vertex Filter

if c - floor(c) == 0
    fprintf(['ERROR: At Iteration #',num2str(n), ', we hit vertex with ID ',num2str(floor(c)),'\n']);
    
    cLog = cLog(1:n);
    ThLog = ThLog(1:n);
    WordLog = WordLog(1:n);

    break
end

cLog(n+1) = c;
ThLog(n+1) = Th;
PtLog(:,n+1) = [xR,yR];
WordLog(n+1) = floor(c);

LogicDebug(1,n) = zeroB;
LogicDebug(2,n) = zeroR;
LogicDebug(3,n) = zeroL;
end % Looper Concludes

disp('Looper Concludes!')
% plot(PtLog(1,:),PtLog(2,:),'Color',DesOrb);

p  = patchline(PtLog(1,:),PtLog(2,:),'linestyle','-','edgecolor','w','linewidth',1,'edgealpha',0.5);

% Normal Lines
quiver(Or0(1),Or0(2),N0(1),N0(2),'Color',DesNorm);
quiver(Or1(1),Or1(2),N1(1),N1(2),'Color',DesNorm);
quiver(Or2(1),Or2(2),N2(1),N2(2),'Color',DesNorm);

hold off;


%% Advanced Angle Corrector


AnAP = AngleA / 2;
AnBP = AngleB / 2 ;

% Filter

% NEW Normals



for k = 1: Iter+1
    if length(WordLog) < Iter+1
        disp('ERROR: Trajectory incomplete, cannot construct BP Map');
        break;
    end
    
    Momentum = [cos(ThLog(k)) sin(ThLog(k))];
    
    if WordLog(k) == 0
      costh = dot(N0,Momentum);
      ThLog(k) = acos(costh);
    elseif WordLog(k) == 1 % Right
         costh = dot(N1,Momentum);
      ThLog(k) = acos(costh);
    elseif WordLog(k) == 2 % Left
      costh = dot(N2,Momentum);
      ThLog(k) = acos(costh);
    end
end

% Angle from Normal 
ThLog = ThLog./pi;

ThLogP = 0.5 - abs(ThLog-0.5);
% Orthogonality Hunter

%% New BP (c-Th) Map with Guides
subplot(3,1,3);
hold on;
grid on;
axis auto;
set(gca,'Color',DesBG);
xlim([0,3]);
ylim([0,0.5]);
plot(cLog,ThLogP,'w.')
plot(cLog(1),ThLogP(1),'co')

% Perimeter Lines
line([1 1], get(gca, 'ylim'),'Color',DesEdge,'LineStyle','--');
line([2 2], get(gca, 'ylim'),'Color',DesEdge,'LineStyle','--');

% Parallel Lines

line([0 3], [0.5 0.5], 'Color',[0.7 0.9 0.3],'LineStyle','-.');

line([0 3], [1/6 1/6], 'Color',[0.7 0.9 0.3],'LineStyle','-.');
line([0 3], [1/4 1/4], 'Color',[0.7 0.9 0.3],'LineStyle','-.');
line([0 3], [1/12 1/12], 'Color',[0.7 0.9 0.3],'LineStyle','-.');
line([0 3], [1/3 1/3], 'Color',[0.7 0.9 0.3],'LineStyle','-.');
line([0 3], [5/12 5/12], 'Color',[0.7 0.9 0.3],'LineStyle','-.');
% Orthogonal Lines

line([0 3], [0 0], 'Color',[0.7 0.9 0.9],'LineStyle','-.');

set(gca,'YTick',[0 0.5],'YTickLabel',{'Orthogonal','Parallel'})
set(gca,'XTick',[0.5 1.5 2.5],'XTickLabel',{'Edge 0 (Base)','Edge 1 (Right)','Edge 2 (Left)'})

hold off;

%% OrbitCollector

PoincareA = [cLog,ThLog];

% Old Attempt
% For (BiOrthogonal Orbits)
iO = find(ThLog==0|ThLog==1, 2, 'first');
if iO<Iter
    disp('Orbit Contains At Least One Orthogonality. Computation Compromised.')
    Period = (iO(2)-iO(1))*2
else
    Simp = uniquetol(PoincareA,1e-7,'ByRows',true);
    Period = length(Simp)
end




%% Downloaded Patchline
function p = patchline(xs,ys,varargin)
% Plot lines as patches (efficiently)
%
% SYNTAX:
%     patchline(xs,ys)
%     patchline(xs,ys,zs,...)
%     patchline(xs,ys,zs,'PropertyName',propertyvalue,...)
%     p = patchline(...)
%
% PROPERTIES: 
%     Accepts all parameter-values accepted by PATCH.
% 
% DESCRIPTION:
%     p = patchline(xs,ys,zs,'PropertyName',propertyvalue,...)
%         Takes a vector of x-values (xs) and a same-sized
%         vector of y-values (ys). z-values (zs) are
%         supported, but optional; if specified, zs must
%         occupy the third input position. Takes all P-V
%         pairs supported by PATCH. Returns in p the handle
%         to the resulting patch object.
%         
% NOTES:
%     Note that we are drawing 0-thickness patches here,
%     represented only by their edges. FACE PROPERTIES WILL
%     NOT NOTICEABLY AFFECT THESE OBJECTS! (Modify the
%     properties of the edges instead.)
%
%     LINUX (UNIX) USERS: One test-user found that this code
%     worked well on his Windows machine, but crashed his
%     Linux box. We traced the problem to an openGL issue;
%     the problem can be fixed by calling 'opengl software'
%     in your <http://www.mathworks.com/help/techdoc/ref/startup.html startup.m>.
%     (That command is valid at startup, but not at runtime,
%     on a unix machine.)
%
% EXAMPLES:
%%% Example 1:
%
% n = 10;
% xs = rand(n,1);
% ys = rand(n,1);
% zs = rand(n,1)*3;
% plot3(xs,ys,zs,'r.')
% xlabel('x');ylabel('y');zlabel('z');
% p  = patchline(xs,ys,zs,'linestyle','--','edgecolor','g',...
%     'linewidth',3,'edgealpha',0.2);
%
%%% Example 2: (Note "hold on" not necessary here!)
%
% t = 0:pi/64:4*pi;
% p(1) = patchline(t,sin(t),'edgecolor','b','linewidth',2,'edgealpha',0.5);
% p(2) = patchline(t,cos(t),'edgecolor','r','linewidth',2,'edgealpha',0.5);
% l = legend('sine(t)','cosine(t)');
% tmp = sort(findobj(l,'type','patch'));
% for ii = 1:numel(tmp)
%     set(tmp(ii),'facecolor',get(p(ii),'edgecolor'),'facealpha',get(p(ii),'edgealpha'),'edgecolor','none')
% end
%
%%% Example 3 (requires Image Processing Toolbox):
%%%   (NOTE that this is NOT the same as showing a transparent image on 
%%%         of the existing image. (That functionality is
%%%         available using showMaskAsOverlay or imoverlay).
%%%         Instead, patchline plots transparent lines over
%%%         the image.)
%
% img = imread('rice.png');
% imshow(img)
% img = imtophat(img,strel('disk',15));
% grains = im2bw(img,graythresh(img));
% grains = bwareaopen(grains,10);
% edges = edge(grains,'canny');
% boundaries = bwboundaries(edges,'noholes');
% cmap = jet(numel(boundaries));
% ind = randperm(numel(boundaries));
% for ii = 1:numel(boundaries)
% patchline(boundaries{ii}(:,2),boundaries{ii}(:,1),...
%     'edgealpha',0.2,'edgecolor',cmap(ind(ii),:),'linewidth',3);
% end
%
% Written by Brett Shoelson, PhD
% brett.shoelson@mathworks.com
% 5/31/2012
% 
% Revisions:
% 6/26 Improved rice.png example, modified FEX image.
%
% Copyright 2012 MathWorks, Inc.
%
% See also: patch, line, plot

[zs,PVs] = parseInputs(varargin{:});
if rem(numel(PVs),2) ~= 0
    % Odd number of inputs!
    error('patchline: Parameter-Values must be entered in valid pairs')
end

% Facecolor = 'k' is (essentially) ignored here, but syntactically necessary
if isempty(zs)
    p = patch([xs(:);NaN],[ys(:);NaN],'k');
else
    p = patch([xs(:);NaN],[ys(:);NaN],[zs(:);NaN],'k');
end

% Apply PV pairs
for ii = 1:2:numel(PVs)
    set(p,PVs{ii},PVs{ii+1})
end
if nargout == 0
    clear p
end
end

function [zs,PVs] = parseInputs(varargin)
if isnumeric(varargin{1})
    zs = varargin{1};
    PVs = varargin(2:end);
else
    PVs = varargin;
    zs = [];
end
end