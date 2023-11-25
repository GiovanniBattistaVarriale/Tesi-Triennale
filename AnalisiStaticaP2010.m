clc; clear; close all;
%% -- Capitolo 1 --
%% Specifiche tecniche
lambda=0.79;
WTO=1200; %peso al decollo kg
W=820; %peso a vuoto kg
Wfuel=200; %peso carburante
Weff=WTO-Wfuel;
rho=1.225; %densità dell'aria a SL
S=13.9; %superficie alare m^2
b=10.3;
carico=WTO/S;
caricoft=carico*0.204817;
Vs=27.78; %velocità di stallo m/s
Vc=33*sqrt(caricoft)*0.51; %velocità di crociera m/s
CLMAX=(2*WTO*9.81)/(rho*S*Vs.^2);
CLMAXr=-CLMAX*40/100;  %CLMAX in volo rovescio
nlim=2.1+24000/(WTO*2.20462+10000);
nlimR=-nlim*40/100; %nlim in volo rovescio
nlim=3.8;
Va=Vs*sqrt(nlim); %velocità di manovra
Vb=sqrt((-2*WTO*9.809)/(rho*CLMAXr*S)); %velocità di stallo in volo rovescio
Vf=Vb*sqrt(-nlimR); %velocità di manovra in volo rovescio
Vd=1.4*Vc; %velocità di picchiata
g=9.81;
c=1.35;
AR=b.^2/S;
e=0.9;
Clalfa=2*pi;
CLalfa=Clalfa/(1+(Clalfa)/(pi*AR*e));

%% -- Capitolo 2 --
%% Diagramma di Manovra

%plot del diagramma
fplot(@(V) 0.5*rho*(V.^2)*(S/(WTO*g))*CLMAX, [0 Vs],'-.k','LineWidth',1); hold on;
fplot(@(V) 0.5*rho*(V.^2)*(S/(WTO*g))*CLMAX, [Vs Va],'-b','LineWidth',2); hold on;
fplot(@(V) 0.5*rho*(V.^2)*(S/(WTO*g))*CLMAXr, [0 Vb],'-.k','LineWidth',1); hold on;
fplot(@(V) 0.5*rho*(V.^2)*(S/(WTO*g))*CLMAXr, [Vb Vf],'-b','LineWidth',2); hold on;
fplot( nlim, [Va Vc],'-b','LineWidth',2); hold on;
fplot( nlim, [Vc Vd],'-b','LineWidth',2); hold on;
fplot( nlimR, [Vf Vc],'-b','LineWidth',2); hold on;

%calcolo n per le vari velocità
ns=0.5*rho*(Vs.^2)*(S/(WTO*g))*CLMAX;
na=0.5*rho*(Va.^2)*(S/(WTO*g))*CLMAX;
nb=0.5*rho*(Vb.^2)*(S/(WTO*g))*CLMAXr;
nf=0.5*rho*(Vf.^2)*(S/(WTO*g))*CLMAXr;
ng=nf;
nc=na;

%marker testuali per i vari punti del diagramma
text(Vs, ns, 'S', 'HorizontalAlignment','center', 'VerticalAlignment','cap','Color','k','FontWeight','bold')
text(Va, na, 'A', 'HorizontalAlignment','center', 'VerticalAlignment','cap','Color','k','FontWeight','bold')
text(Vd, na, 'D', 'HorizontalAlignment','center', 'VerticalAlignment','cap','Color','k','FontWeight','bold')
text(Vb, nb, 'B', 'HorizontalAlignment','center', 'VerticalAlignment','cap','Color','k','FontWeight','bold')
text(Vf, nf, 'F', 'HorizontalAlignment','center', 'VerticalAlignment','cap','Color','k','FontWeight','bold')
text(Vc, ng, 'G', 'HorizontalAlignment','center', 'VerticalAlignment','cap','Color','k','FontWeight','bold')
text(Vd, 0, 'E', 'HorizontalAlignment','center', 'VerticalAlignment','cap','Color','k','FontWeight','bold')
text(Vc, nc, 'C', 'HorizontalAlignment','center', 'VerticalAlignment','cap','Color','k','FontWeight','bold')

%tratteggio
fplot( ns,[0,Vs],'--k');
fplot( nb,[0,Vb],'--k');
fplot( na,[0,Va],'--k');
fplot( nf,[0,Vf],'--k');
fplot( nf,[Vc,Vd],'--k');
fplot( 0,[0,Vd],'-.r');
xline([Vs,Vd,Va,Vb,Vc],'--');


%interpolazione per chiudere il diagramma
D = [Vd,na];
E = [Vd,0];
G= [Vc,ng];
x = [D(1) E(1)];
y = [D(2) E(2)];
plot(x,y,'b','LineWidth',2);
hold on;
x = [G(1) E(1)];
y = [G(2) E(2)];
plot(x,y,'b','LineWidth',2);
hold on;

%titolo e assi
title('Diagramma di manovra');
xlabel('V (m/s)');ylabel('n');
grid on;
grid minor;
axis([0 110 -2.5 4.5]);

%% Diagramma di Raffica
mu_g=(2*(carico*g)/(rho*g*c*CLalfa)); %fattore di attenuazione di raffica
Kg=(0.88*mu_g)/(5.83+mu_g); %parametro di massa

U_Vc=15.24;
U_Vd=7.62;

ng_Vc=1+(Kg*rho*U_Vc*CLalfa*Vc)/(2*(carico*g));
ng_Vd=1+(Kg*rho*U_Vd*CLalfa*Vd)/(2*(carico*g));
ngr_Vc=1-(Kg*rho*U_Vc*CLalfa*Vc)/(2*(carico*g));
ngr_Vd=1-(Kg*rho*U_Vd*CLalfa*Vd)/(2*(carico*g));

figure(2);
X = [Vc,ng_Vc];
K = [Vc,ngr_Vc];
Y= [Vd,ng_Vd];
Z= [Vd,ngr_Vd];
O= [0,1];
x = [O(1) X(1)];
y = [O(2) X(2)];
plot(x,y,'r','LineWidth',2);
hold on;
x = [O(1) Y(1)];
y = [O(2) Y(2)];
plot(x,y,'r','LineWidth',2);
hold on;
x = [O(1) Z(1)];
y = [O(2) Z(2)];
plot(x,y,'r','LineWidth',2);
hold on;
x = [O(1) K(1)];
y = [O(2) K(2)];
plot(x,y,'r','LineWidth',2);
hold on;

x = [X(1) Y(1)];
y = [X(2) Y(2)];
plot(x,y,'r','LineWidth',2);
hold on;
x = [Y(1) Z(1)];
y = [Y(2) Z(2)];
plot(x,y,'r','LineWidth',2);
hold on;
x = [K(1) Z(1)];
y = [K(2) Z(2)];
plot(x,y,'r','LineWidth',2);
hold on;

fplot( 0,[0,Vd],'-.k');


%titolo e assi
title('Diagramma di raffica');
xlabel('V (m/s)');ylabel('n');
grid on;
grid minor;
axis([0 110 -2.5 4.5]);


%% Inviluppo di volo
figure(3);
X = [Vc,ng_Vc];
K = [Vc,ngr_Vc];
Y= [Vd,ng_Vd];
Z= [Vd,ngr_Vd];
O= [0,1];
x = [O(1) X(1)];
y = [O(2) X(2)];
plot(x,y,'r','LineWidth',1.5);
hold on;
x = [O(1) Y(1)];
y = [O(2) Y(2)];
plot(x,y,'r','LineWidth',1.5);
hold on;
x = [O(1) Z(1)];
y = [O(2) Z(2)];
plot(x,y,'r','LineWidth',1.5);
hold on;
x = [O(1) K(1)];
y = [O(2) K(2)];
plot(x,y,'r','LineWidth',1.5);
hold on;

x = [X(1) Y(1)];
y = [X(2) Y(2)];
plot(x,y,'r','LineWidth',1.5);
hold on;
x = [Y(1) Z(1)];
y = [Y(2) Z(2)];
plot(x,y,'r','LineWidth',1.5);
hold on;
x = [K(1) Z(1)];
y = [K(2) Z(2)];
plot(x,y,'r','LineWidth',1.5);
hold on;

fplot( 0,[0,Vd],'-.k');


%plot del diagramma
fplot(@(V) 0.5*rho*(V.^2)*(S/(WTO*g))*CLMAX, [0 Vs],'-.k','LineWidth',1); hold on;
fplot(@(V) 0.5*rho*(V.^2)*(S/(WTO*g))*CLMAX, [Vs Va],'-b','LineWidth',2); hold on;
fplot(@(V) 0.5*rho*(V.^2)*(S/(WTO*g))*CLMAXr, [0 Vb],'-.k','LineWidth',1); hold on;
fplot(@(V) 0.5*rho*(V.^2)*(S/(WTO*g))*CLMAXr, [Vb Vf],'-b','LineWidth',2); hold on;
fplot( nlim, [Va Vc],'-b','LineWidth',2); hold on;
fplot( nlim, [Vc Vd],'-b','LineWidth',2); hold on;
fplot( nlimR, [Vf Vc],'-b','LineWidth',2); hold on;

%calcolo n per le vari velocità
ns=0.5*rho*(Vs.^2)*(S/(WTO*g))*CLMAX;
na=0.5*rho*(Va.^2)*(S/(WTO*g))*CLMAX;
nb=0.5*rho*(Vb.^2)*(S/(WTO*g))*CLMAXr;
nf=0.5*rho*(Vf.^2)*(S/(WTO*g))*CLMAXr;
ng=nf;
nc=na;

%marker testuali per i vari punti del diagramma
text(Vs, ns, 'S', 'HorizontalAlignment','center', 'VerticalAlignment','cap','Color','k','FontWeight','bold')
text(Va, na, 'A', 'HorizontalAlignment','center', 'VerticalAlignment','cap','Color','k','FontWeight','bold')
text(Vd, na, 'D', 'HorizontalAlignment','center', 'VerticalAlignment','cap','Color','k','FontWeight','bold')
text(Vb, nb, 'B', 'HorizontalAlignment','center', 'VerticalAlignment','cap','Color','k','FontWeight','bold')
text(Vf, nf, 'F', 'HorizontalAlignment','center', 'VerticalAlignment','cap','Color','k','FontWeight','bold')
text(Vc, ng, 'G', 'HorizontalAlignment','center', 'VerticalAlignment','cap','Color','k','FontWeight','bold')
text(Vd, 0, 'E', 'HorizontalAlignment','center', 'VerticalAlignment','cap','Color','k','FontWeight','bold')
text(Vc, nc, 'C', 'HorizontalAlignment','center', 'VerticalAlignment','cap','Color','k','FontWeight','bold')

%tratteggio
fplot( ns,[0,Vs],'--k');
fplot( nb,[0,Vb],'--k');
fplot( na,[0,Va],'--k');
fplot( nf,[0,Vf],'--k');
fplot( nf,[Vc,Vd],'--k');
fplot( 0,[0,Vd],'-.r');
xline([Vs,Vd,Va,Vb,Vc],'--');


%interpolazione per chiudere il diagramma
D = [Vd,na];
E = [Vd,0];
G= [Vc,ng];
x = [D(1) E(1)];
y = [D(2) E(2)];
plot(x,y,'b','LineWidth',2);
hold on;
x = [G(1) E(1)];
y = [G(2) E(2)];
plot(x,y,'b','LineWidth',2);
hold on;


%titolo e assi
title('Inviluppo di volo');
xlabel('V (m/s)');ylabel('n');
grid on;
grid minor;
axis([0 110 -2.5 4.5]);

%% -- Capitolo 3 --
%% Calcolo P e L
c=1.29; %corda media aerodinamica
P0=0;
P=P0;
it=0;
err=100000;
n=nlim;
L=n*Weff*g-P;
CMF=-0.085; 
M0D=0.5*rho*(Vd.^2)*S*c*abs(CMF);

a=c*25/100; 
l=4.6; 


disp('Calcolo P e L');
disp(['Iterazione ',num2str(it), ' L: ', num2str(L),' P: ', num2str(P),' errore: ',num2str(err)])
while abs(err)>0.00001
    it=it+1;
    L=n*Weff*g-P;
    Pold=P;
    P=(n*Weff*g*a-M0D)/(l+a);
    err=abs(Pold-P);
    disp(['Iterazione ',num2str(it), ' L: ', num2str(L),' P: ', num2str(P),' errore: ',num2str(err)])
end

CLw=L/(0.5*rho*(Vd.^2)*S);
Ww=g*0.15*WTO;
Whf=Ww/2;
qw=Whf*4/b;



fprintf(['\nCLw: ',num2str(CLw),'\nWw: ',num2str(Ww),' N\n\n']);

%% Stima del carico - Anderson

%AR=7.63

La0x=1.203;
La0y=1.198;

La2x=1.174;
La2y=1.168;

La4x=1.120;
La4y=1.113;

La6x=1.024;
La6y=1.024;

La8x=0.845;
La8y=0.854;

La9x=0.660;
La9y=0.673;

La95x=0.502;
La95y=0.521;

La975x=0.366;
La975y=0.383;

%interpolazione valori di La
La0=(La0x)*0.3+(La0y*0.7);
La2=(La2x)*0.3+(La2y*0.7);
La4=(La4x)*0.3+(La4y*0.7);
La6=(La6x)*0.3+(La6y*0.7);
La8=(La8x)*0.3+(La8y*0.7);
La9=(La9x)*0.3+(La9y*0.7);
La95=(La95x)*0.3+(La95y*0.7);
La975=(La975x)*0.3+(La975y*0.7);

c=1.3;
c0=c;
Cla0=S*La0/(c*b);
Cl0=Cla0*CLw;
l0=La0*L/b;
q0=l0-nlim*qw*(1-0);

c=1.4;
c2=c;
Cla2=S*La2/(c*b);
Cl2=Cla2*CLw;
l2=La2*L/b;
q2=l2-nlim*qw*(1-0.2);

c4=c;
Cla4=S*La4/(c*b);
Cl4=Cla4*CLw;
l4=La4*L/b;
q4=l4-nlim*qw*(1-0.4);

c6=c;
Cla6=S*La6/(c*b);
Cl6=Cla6*CLw;
l6=La6*L/b;
q6=l6-nlim*qw*(1-0.6);

c=1.4-((1.4-1.1)*0.2);
c8=c;
Cla8=S*La8/(c*b);
Cl8=Cla8*CLw;
l8=La8*L/b;
q8=l8-nlim*qw*(1-0.8);

c=1.4-((1.4-1.1)*0.3);
c9=c;
Cla9=S*La9/(c*b);
Cl9=Cla9*CLw;
l9=La9*L/b;
q9=l9-nlim*qw*(1-0.9);

c=1.4-((1.4-1.1)*0.35);
c95=c;
Cla95=S*La95/(c*b);
Cl95=Cla95*CLw;
l95=La95*L/b;
q95=l95-nlim*qw*(1-0.95);

c=1.4-((1.4-1.1)*0.375);
c975=c;
Cla975=S*La975/(c*b);
Cl975=Cla975*CLw;
l975=La975*L/b;
q975=l975-nlim*qw*(1-0.975);




fprintf(['eta','\tLa      ','\tc     ','\tCla     ','\tCl     ','\tl     ','\tq eff    ','\n']);
fprintf(['0\t',num2str(La0),'\t ',num2str(c0),'\t ',num2str(Cla0),'\t ',num2str(Cl0),'\t ',num2str(l0),'\t ',num2str(q0),'\n']);
fprintf(['0.2\t',num2str(La2),'\t ',num2str(c2),'\t ',num2str(Cla2),'\t ',num2str(Cl2),'\t ',num2str(l2),'\t ',num2str(q2),'\n']);
fprintf(['0.4\t',num2str(La4),'\t ',num2str(c4),'\t ',num2str(Cla4),'\t ',num2str(Cl4),'\t ',num2str(l4),'\t ',num2str(q4),'\n']);
fprintf(['0.6\t',num2str(La6),'\t ',num2str(c6),'\t ',num2str(Cla6),'\t ',num2str(Cl6),'\t ',num2str(l6),'\t ',num2str(q6),'\n']);
fprintf(['0.8\t',num2str(La8),'\t ',num2str(c8),'\t ',num2str(Cla8),'\t ',num2str(Cl8),'\t ',num2str(l8),'\t ',num2str(q8),'\n']);
fprintf(['0.9\t',num2str(La9),'\t ',num2str(c9),'\t ',num2str(Cla9),'\t ',num2str(Cl9),'\t ',num2str(l9),'\t ',num2str(q9),'\n']);
fprintf(['0.95\t',num2str(La95),'\t ',num2str(c95),'\t ',num2str(Cla95),'\t ',num2str(Cl95),'\t ',num2str(l95),'\t ',num2str(q95),'\n']);
fprintf(['0.975\t',num2str(La975),'\t ',num2str(c975),'\t ',num2str(Cla975),'\t ',num2str(Cl975),'\t ',num2str(l975),'\t ',num2str(q975),'\n']);


%% grafica anderson

figure(4)
eta=[0 0.2 0.4 0.6 0.8 0.9 0.95 0.975 1];
q=[q0 q2 q4 q6 q8 q9 q95 q975 0];

labels = {q0 q2 q4 q6 q8 q9 q95 q975 0};

xx=linspace(eta(1),eta(end),1000);
yy=interp1(eta,q,xx,'spline');  
plot(xx,yy,eta,q,'.','MarkerSize',22);

text(eta,q,labels,'VerticalAlignment','bottom','HorizontalAlignment','left');


xlabel('\eta'); ylabel('Carico effettivo (N/m)');
title("Distribuzione carico punto D");
grid on;


%% sollecitazione in assenza di controvento
eta=flip(eta);
q=flip(q);
bdim=eta.*b/2;
T=zeros(9,1);
T(1)=0;
for it=1:8
    T(it+1)=(q(it+1)+q(it))*(bdim(it)-bdim(it+1))*0.5+T(it);
end

figure(5);
subplot(2,1,1);
labels = {T(1) T(2) T(3) T(4) T(5) T(6) T(7) T(8) T(9)};
plot(bdim,T,'-',bdim,T,'.','MarkerSize',22);
text(bdim,T,labels,'VerticalAlignment','bottom','HorizontalAlignment','left');
xlabel('y (m)'); ylabel('Taglio (N)');
title("Punto D - Taglio senza controvento");

grid on;

%momento 

fun = @(bdim) -T(9)*(1-bdim/5.15);
subplot(2,1,2);
M=cumtrapz(bdim,fun(bdim));

plot(bdim,M,'-',bdim,M,'.','MarkerSize',22);
labels = {M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8) M(9)};
plot(bdim,M,'-',bdim,M,'.','MarkerSize',22);
text(bdim,M,labels,'VerticalAlignment','bottom','HorizontalAlignment','left');
xlabel('y (m)'); ylabel('Momento (Nm)');
title("Punto D - Momento senza controvento");

grid on;

%% sollecitazione con controvento
Dcontrovento=2.15; %distanza controvento dalla radice
Rb=M(9)/Dcontrovento; %Reazione del controvento

T=nan(9,1);
T(1)=0;
for it=1:6
    T(it+1)=(q(it+1)+q(it))*(bdim(it)-bdim(it+1))*0.5+T(it);
end
A=[bdim(7),T(7)];


figure(6);
subplot(2,1,1);
labels = {T(1) T(2) T(3) T(4) T(5) T(6) T(7) T(8) T(9)};
plot(bdim,T,'-b',bdim,T,'.r','MarkerSize',22); hold on;
text(bdim,T,labels,'VerticalAlignment','bottom','HorizontalAlignment','left');
xlabel('y (m)'); ylabel('Taglio (N)');
title("Punto D - Taglio (con controvento)");
text(5,7000,["Rb: ",num2str(Rb)]);
grid on;
T(1:6)=nan;
TS(1)=T(7);
T(7)=T(7)-Rb;
TS(2)=T(7);
Tmax=min((TS));
text(0.1,7000,["Taglio max(N): ",num2str(Tmax)]);
for it=7:8
    T(it+1)=(q(it+1)+q(it))*(bdim(it)-bdim(it+1))*0.5+T(it);
end
plot(bdim,T,'-b',bdim,T,'.r','MarkerSize',22); hold on;
B=[bdim(7),T(7)];
x = [A(1) B(1)];
y = [A(2) B(2)];
plot(x,y,'-b');
hold on;


M=zeros(9,1);
M(1)=0;
T=nan(9,1);
T(1)=0;
for it=1:6
    T(it+1)=(q(it+1)+q(it))*(bdim(it)-bdim(it+1))*0.5+T(it);
end

for it=1:8
    M(it+1)=(T(it+1)+T(it))*(bdim(it)-bdim(it+1))*0.5+M(it);
end
T(1:6)=nan;
T(7)=T(7)-Rb;
for it=7:8
    T(it+1)=(q(it+1)+q(it))*(bdim(it)-bdim(it+1))*0.5+T(it);
end
for it=7:8
    M(it+1)=(T(it+1)+T(it))*(bdim(it)-bdim(it+1))*0.5+M(it);
end
Mmax=M(7);
M(9)=0; 
subplot(2,1,2);
labels = {M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8) M(9)};
plot(bdim,M,'-',bdim,M,'.','MarkerSize',22);
text(0.1,10000,["Momento max(N*M): ",num2str(M(7))]);
text(bdim,M,labels,'VerticalAlignment','bottom','HorizontalAlignment','left');
xlabel('y (m)'); ylabel('Momento (Nm)');
title("Punto D - Momento (con controvento)");
grid on;


%% Torsione
c=1.29;
Vg=Vc;
Wnf=(WTO-Wfuel)*9.81;
M0=-1/2*rho*(Vd^2)*S*c*CMF;
MtD=M0/2;

M0=-1/2*rho*(Vg^2)*S*c*CMF;
L=Wnf*ngr_Vc;
MtG=M0/2-L*a/2;

Mt=max(MtD,MtG); %punto più critico è G(punto C volo rovescio)

P=(ngr_Vc*a*Wnf-M0)/(l+a);
L=Wnf*ngr_Vc-P;
CLw=L/(0.5*rho*(Vc^2)*S);

%% Torsione- carico e grafica carico
c=1.3;
c0=c;
Cla0=S*La0/(c*b);
Cl0=Cla0*CLw;
l0=La0*L/b;
q0=l0-nlim*qw*(1-0);

c=1.4;
c2=c;
Cla2=S*La2/(c*b);
Cl2=Cla2*CLw;
l2=La2*L/b;
q2=l2-nlim*qw*(1-0.2);

c4=c;
Cla4=S*La4/(c*b);
Cl4=Cla4*CLw;
l4=La4*L/b;
q4=l4-nlim*qw*(1-0.4);

c6=c;
Cla6=S*La6/(c*b);
Cl6=Cla6*CLw;
l6=La6*L/b;
q6=l6-nlim*qw*(1-0.6);

c=1.4-((1.4-1.1)*0.2);
c8=c;
Cla8=S*La8/(c*b);
Cl8=Cla8*CLw;
l8=La8*L/b;
q8=l8-nlim*qw*(1-0.8);

c=1.4-((1.4-1.1)*0.3);
c9=c;
Cla9=S*La9/(c*b);
Cl9=Cla9*CLw;
l9=La9*L/b;
q9=l9-nlim*qw*(1-0.9);

c=1.4-((1.4-1.1)*0.35);
c95=c;
Cla95=S*La95/(c*b);
Cl95=Cla95*CLw;
l95=La95*L/b;
q95=l95-nlim*qw*(1-0.95);

c=1.4-((1.4-1.1)*0.375);
c975=c;
Cla975=S*La975/(c*b);
Cl975=Cla975*CLw;
l975=La975*L/b;
q975=l975-nlim*qw*(1-0.975);

fprintf(['eta','\tLa      ','\tc     ','\tCla     ','\tCl     ','\tl     ','\tq eff    ','\n']);
fprintf(['0\t',num2str(La0),'\t ',num2str(c0),'\t ',num2str(Cla0),'\t ',num2str(Cl0),'\t ',num2str(l0),'\t ',num2str(q0),'\n']);
fprintf(['0.2\t',num2str(La2),'\t ',num2str(c2),'\t ',num2str(Cla2),'\t ',num2str(Cl2),'\t ',num2str(l2),'\t ',num2str(q2),'\n']);
fprintf(['0.4\t',num2str(La4),'\t ',num2str(c4),'\t ',num2str(Cla4),'\t ',num2str(Cl4),'\t ',num2str(l4),'\t ',num2str(q4),'\n']);
fprintf(['0.6\t',num2str(La6),'\t ',num2str(c6),'\t ',num2str(Cla6),'\t ',num2str(Cl6),'\t ',num2str(l6),'\t ',num2str(q6),'\n']);
fprintf(['0.8\t',num2str(La8),'\t ',num2str(c8),'\t ',num2str(Cla8),'\t ',num2str(Cl8),'\t ',num2str(l8),'\t ',num2str(q8),'\n']);
fprintf(['0.9\t',num2str(La9),'\t ',num2str(c9),'\t ',num2str(Cla9),'\t ',num2str(Cl9),'\t ',num2str(l9),'\t ',num2str(q9),'\n']);
fprintf(['0.95\t',num2str(La95),'\t ',num2str(c95),'\t ',num2str(Cla95),'\t ',num2str(Cl95),'\t ',num2str(l95),'\t ',num2str(q95),'\n']);
fprintf(['0.975\t',num2str(La975),'\t ',num2str(c975),'\t ',num2str(Cla975),'\t ',num2str(Cl975),'\t ',num2str(l975),'\t ',num2str(q975),'\n\n']);




figure(7)
eta=[0 0.2 0.4 0.6 0.8 0.9 0.95 0.975 1];
q=[q0 q2 q4 q6 q8 q9 q95 q975 0];

labels = {q0 q2 q4 q6 q8 q9 q95 q975 0};

xx=linspace(eta(1),eta(end),1000);
yy=interp1(eta,q,xx,'spline');  
plot(xx,yy,eta,q,'.','MarkerSize',22);

text(eta,q,labels,'VerticalAlignment','bottom','HorizontalAlignment','left');


xlabel('\eta'); ylabel('Carico effettivo (N/m)');
title("Distribuzione carico punto G");
grid on;

%% Torsione-sollecitazione in assenza di controvento
eta=flip(eta);
q=flip(q);
bdim=eta.*b/2;
T=zeros(9,1);
T(1)=0;
for it=1:8
    T(it+1)=(q(it+1)+q(it))*(bdim(it)-bdim(it+1))*0.5+T(it);
end


figure(8);
subplot(2,1,1);
labels = {T(1) T(2) T(3) T(4) T(5) T(6) T(7) T(8) T(9)};
plot(bdim,T,'-',bdim,T,'.','MarkerSize',22);
text(bdim,T,labels,'VerticalAlignment','bottom','HorizontalAlignment','left');
xlabel('y (m)'); ylabel('Taglio (N)');
title("Punto G - Taglio senza controvento");

grid on;

%momento

fun = @(bdim) -T(9)*(1-bdim/5.15);
subplot(2,1,2);
M=cumtrapz(bdim,fun(bdim));

subplot(2,1,2);
labels = {M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8) M(9)};
plot(bdim,M,'-',bdim,M,'.','MarkerSize',22);
text(bdim,M,labels,'VerticalAlignment','bottom','HorizontalAlignment','left');
xlabel('y (m)'); ylabel('Momento (Nm)');
title("Punto G - Momento senza controvento");

grid on;

%% Torsione- controvento
Dcontrovento=2.15; %distanza controvento dalla radice
Rb=M(9)/Dcontrovento; %Reazione del controvento
T=nan(9,1);
T(1)=0;
for it=1:6
    T(it+1)=(q(it+1)+q(it))*(bdim(it)-bdim(it+1))*0.5+T(it);
end
A=[bdim(7),T(7)];


figure(9);
subplot(2,1,1);
labels = {T(1) T(2) T(3) T(4) T(5) T(6) T(7) T(8) T(9)};
plot(bdim,T,'-b',bdim,T,'.r','MarkerSize',22); hold on;
text(bdim,T,labels,'VerticalAlignment','bottom','HorizontalAlignment','left');
xlabel('y (m)'); ylabel('Taglio (N)');
title("Punto G - Taglio");
text(5,7000,["Rb: ",num2str(Rb)]);
grid on;
    
    
T(1:6)=nan;
TS(1)=T(7);
T(7)=T(7)-Rb;
TS(2)=T(7);
Tmax_Torsione=max((TS));
text(0.1,7000,["Taglio max(N): ",num2str(Tmax_Torsione)]);
for it=7:8
    T(it+1)=(q(it+1)+q(it))*(bdim(it)-bdim(it+1))*0.5+T(it);
end
plot(bdim,T,'-b',bdim,T,'.r','MarkerSize',22); hold on;
B=[bdim(7),T(7)];
x = [A(1) B(1)];
y = [A(2) B(2)];
plot(x,y,'-b');
hold on;
   



M=zeros(9,1);
M(1)=0;
T=nan(9,1);
T(1)=0;
for it=1:7
    T(it+1)=(q(it+1)+q(it))*(bdim(it)-bdim(it+1))*0.5+T(it);
end

for it=1:7
    M(it+1)=(T(it+1)+T(it))*(bdim(it)-bdim(it+1))*0.5+M(it);
end
T(1:6)=nan;
T(7)=T(7)-Rb;
for it=7:8
    T(it+1)=(q(it+1)+q(it))*(bdim(it)-bdim(it+1))*0.5+T(it);
end
for it=7:8
    M(it+1)=(T(it+1)+T(it))*(bdim(it)-bdim(it+1))*0.5+M(it);
end
M(9)=0; 
subplot(2,1,2);
labels = {M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8) M(9)};
plot(bdim,M,'-',bdim,M,'.','MarkerSize',22); hold on;
text(0.1,0,["Momento max(N*M): ",num2str(M(7))]);
text(bdim,M,labels,'VerticalAlignment','bottom','HorizontalAlignment','left');
xlabel('y (m)'); ylabel('Momento (Nm)');
title("Punto G - Momento");

grid on;

%% Torsione- Determinazione stato tensionale
Mmax_Torsione=M(7); %calcolare flussi di taglio

%% -- Capitolo 4 --
%% Profilo 631 A012
%pagina 344
c=1.29;
x=[0 0.5 0.75 1.25 2.5 5.0 7.5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100];
y=[0 0.973 1.173 1.492 2.078 2.895 3.504 3.994 4.747 5.287 5.664 5.901 5.995 5.957 5.792 5.517 5.148 4.7 4.186 3.621 3.026 2.426 1.826 1.225 0.625 0.025];
x1=x./100;
y1=y./100;
figure(10);
plot(x1,y1); hold on;plot(x1,-y1);
axis([0 1 -0.5 0.5]);grid on;
xlabel('x/c'); ylabel('y/c');
title("NACA 63(1)A012 (adimensionale)");

%non adimensionale
x=x.*c/100;
y=y.*c/100;
figure(11);
up=plot(x,y); hold on; low=plot(x,-y);hold on;
axis([0 1.29 -0.5 0.5]);grid on;
xlabel('x(m)'); ylabel('y(m)');
title("NACA 63(1)A012 (c=1.29m)");

xline(c*25/100,'-.');
xline(c*70/100,'-.');
x20=abs(y(11)+y(11)); % dal 25% al 70% della cma
x70=abs(y(20)+y(20));

%wingbox
figure(12)
solettex=[25  35  45  55   70].*c/100;
solettey=[5.664 5.995 5.792 5.148 3.621]*c/100;
labels={'1','2','3','4','5'};
top=plot(solettex,solettey,'-ok','LineWidth',1.5,'MarkerSize',5); hold on;
text(solettex,solettey,labels,'VerticalAlignment','bottom','HorizontalAlignment','right');
labels={'10','9','8','7','6'};
bot=plot(solettex,-solettey,'-ok','LineWidth',1.5,'MarkerSize',5); hold on;
text(solettex,-solettey,labels,'VerticalAlignment','bottom','HorizontalAlignment','right');
axis([0 1.29 -0.5 0.5]);grid on;
xlabel('x(m)'); ylabel('y(m)');
title("NACA 63(1)A012 Wingbox");
as=[solettex(1),solettex(1)];
bs=[solettey(1),-solettey(1)];
plot(as,bs,'-k','LineWidth',1.5);
as=[solettex(5),solettex(5)];
bs=[solettey(5),-solettey(5)];
plot(as,bs,'-k','LineWidth',1.5);


x=[0 0.5 0.75 1.25 2.5 5.0 7.5 10 15 20 25].*c/100;
y=[0 0.973 1.173 1.492 2.078 2.895 3.504 3.994 4.747 5.287 5.664].*c/100;
plot(x,y,'k','LineWidth',1.5); hold on;
plot(x,-y,'k','LineWidth',1.5); hold on;
xline(0.3225,'--');
yline(0,'--');

%wingbox sistema di riferimento per i flussi di taglio
figure(13)
solettex=[25  35  45  55   70].*c/100;
solettey=[5.664 5.995 5.792 5.148 3.621]*c/100;
labels={'1','2','3','4','5'};
top=plot(solettex,solettey,'-ok','LineWidth',1.5,'MarkerSize',5); hold on;
text(solettex,solettey,labels,'VerticalAlignment','bottom','HorizontalAlignment','right');
labels={'10','9','8','7','6'};
bot=plot(solettex,-solettey,'-ok','LineWidth',1.5,'MarkerSize',5); hold on;
text(solettex,-solettey,labels,'VerticalAlignment','bottom','HorizontalAlignment','right');
axis([-0.2 1.29 -0.5 0.5]);
xlabel('x(m)'); ylabel('y(m)');
title("NACA 63(1)A012 Wingbox, no Grid");
as=[solettex(1),solettex(1)];
bs=[solettey(1),-solettey(1)];
plot(as,bs,'-k','LineWidth',1.5);
as=[solettex(5),solettex(5)];
bs=[solettey(5),-solettey(5)];
plot(as,bs,'-k','LineWidth',1.5);


x=[0 0.5 0.75 1.25 2.5 5.0 7.5 10 15 20 25].*c/100;
y=[0 0.973 1.173 1.492 2.078 2.895 3.504 3.994 4.747 5.287 5.664].*c/100;
plot(x,y,'k','LineWidth',1.5); hold on;
plot(x,-y,'k','LineWidth',1.5); hold on;



%% Wingbox
Ce=0.342; %forma corrente
h1=x20*1000; 
h2=x70*1000;
Lwingbox=(70-25)*c/100;
a=20;
l=14;
L=200; %larghezza pannello
d=440; %lunghezza pannello
ncorr=5; %numero correnti
tp=0.6; %spessore pannello
b=(d/ncorr-1); %baia
E=71.7*1000;
sigmacy=460;
ni=0.3;
T=Tmax;
tc=0.8;
sigmacs=(E*sigmacy)^0.5*Ce*((a+l)/(2*tc))^-0.75;
ta=0.9; %spessore anima

coeff_ks=h1/b;
tau=abs(T)/(h1*ta);
ks=7;
taucr=(ks*(pi.^2)*E*ta.^2)/(12*(1-ni.^2)*b^2);
tsut=tau/taucr;
k=tanh(0.5*log(tsut)); %si vuole k circa 0.4 0.5
ktensione=k;

c=16*tp;
Acollab=2*c*tp;
A2=2*a*l;
Ac=2*a*l-(2*a-2*tc)*(l-tc);
Atot=Ac+Acollab;
%mom statici
S1=2*c*tp*tp/2;
S2=2*a*l*((l/2+tp));
S3=(2*a-2*tc)*(l-tc)*(tp+tc+l/2-tc/2);
Stot=S1+S2-S3;
rG=Stot/Atot;
I1G=(2*c*tp^3)/12+2*c*tp*(rG-tp/2)^2;
I2G=(2*a*l^3)/12+2*a*l*(l/2+tp-rG)^2;
I3G=((2*a-2*tc)*(l-tc)^3)/12+(2*a-2*tc)*(l-tc)*(tp+tc/2+l/2-rG)^2;
ItotG=I1G+I2G-I3G;
rhomin=sqrt(ItotG/Atot);
lambda=L/rhomin; %L0=L perchè simply supported
lambda_lim=pi*sqrt(2*E/sigmacs);
%lamba<lamba_lim
sigmacrCorrente=sigmacs-(sigmacs^2)*lambda^2/(4*(pi^2)*E);
%prima stima del carico
Pcr=sigmacrCorrente*(ncorr*Ac+2*c*tp*(ncorr-1));
rapporto_per_Kc=L/b;
m=2; kc=(m*b/L+L/(b*m))^2;
sigmacrp=((kc*E*pi^2)/(12*(1-ni^2)))*(tp/b)^2;
c2=b*(1+sigmacrp/sigmacrCorrente)/4;
errore=(c2-c)/c2;
cold=c2;
%successive iterazioni
    it=1;
    fprintf('\nCarico Critico\n');
    fprintf(['iterazione: ',num2str(it),' Pcr: ',num2str(Pcr),'\n']);
while errore>0.0000001
    it=it+1;
Acollab=2*cold*tp;
Atot=Ac+Acollab;

S1=2*cold*tp*tp/2;
Stot=S1+S2-S3;
rG=Stot/Atot;

I1G=(2*cold*tp^3)/12+2*cold*tp*(rG-tp/2)^2;
I2G=(2*a*l^3)/12+2*a*l*(l/2+tp-rG)^2;
I3G=((2*a-2*tc)*(l-tc)^3)/12+(2*a-2*tc)*(l-tc)*(tp+tc/2+l/2-rG)^2;
ItotG=I1G+I2G-I3G;

rhomin=sqrt(ItotG/Atot);
lambda=L/rhomin; 

sigmacrCorrente2=sigmacs-(sigmacs^2)*lambda^2/(4*(pi^2)*E);

Pcr=sigmacrCorrente2*(ncorr*Ac+2*cold*tp*(ncorr-1));
sigmacrp2=((kc*E*pi^2)/(12*(1-ni^2)))*(tp/b)^2;

cit2=b*(1+sigmacrp2/sigmacrCorrente2)/4;
errore=(cit2-cold)/cit2;
cold=cit2;
fprintf(['iterazione: ',num2str(it),' Pcr: ',num2str(Pcr),' Errore: ',num2str(errore),'\n']);
end

%% aree solette
Mx=Mmax;
br=(h1+h2)/2;
br=br/1000;
Mp=Pcr*br;
DeltaM=abs(Mx-Mp); %momento effettivo assorbito dalle solette
zmax=h1/2;
sigmamax=150/1000; %controllare
Ilong=DeltaM*zmax/sigmamax;
Iprinc=0.7*Ilong;
Aprinc=4*Iprinc/(2*h1^2);
Isec=0.3*Ilong;
Asec=(4*Isec)/(2*h2^2);
fprintf('\nAree solette\n');
fprintf(['Ilong: ',num2str(Ilong),' Iprinc: ',num2str(Iprinc),' Aprinc: ',num2str(Aprinc),'\nIsec: ',num2str(Isec),' Asec: ',num2str(Asec),' Atot: ',num2str(Atot),'\n\n'])

ASol=zeros(10,1);
ASol(1:10)=Atot;
ASol(5)=Asec;
ASol(6)=ASol(5);
ASol(1)=Aprinc;
ASol(10)=ASol(1);

dist_da_y=Lwingbox/4;

%distanze solette rispetto all'asse y
dy=zeros(10,1);
dy(1)=0; dy(10)=0;
dy(2)=dist_da_y; dy(9)=dy(2);
dy(3)=2*dist_da_y; dy(8)=dy(3);
dy(4)=3*dist_da_y; dy(7)=dy(4);
dy(5)=4*dist_da_y; dy(6)=dy(5);
%distanze solette rispetto all'asse x
dx=zeros(10,1);
dx(1:5)=solettey;
dx(6:10)=-flip(solettey);

Atot=sum(ASol,'all');

%% baricentro
it=0;
xg=0;
for it=1:10
xg=xg+ASol(it)*dy(it);
it=it+1;
end
it=0;
yg=0;

for it=1:10
yg=yg+ASol(it)*dx(it);
it=it+1;
end
xg=xg/Atot;
yg=yg/Atot; %uguale a zero perchè simmetrica
c=1.29;
plot(xg+25*c/100,yg,'or');

dx=dx.*1000;
dy=dy.*1000;
xg=xg*1000;
%% momenti di inerzia
Ixy=0; %profilo simmetrico
Ixx=0;
Iyy=0;
it=1;
for it=1:10
Ixx=Ixx+ASol(it)*(dx(it))^2;
it=it+1;
end
it=1;
for it=1:10;
Iyy=Iyy+ASol(it)*(dy(it)-xg)^2;
it=it+1;
end

%% calcolo delle sigma
Mx=-Mmax*1000;
sigmazeta_su_y=(Mx/(Ixx));
it=0;
sigmazeta(10)=zeros;
for it=1:10
sigmazeta(it)=sigmazeta_su_y*dx(it);
fprintf(['Sigma ',num2str(it),'= ',num2str(sigmazeta(it)),'MPa\n']);
it=it+1;
end


%% calcolo qb
it=0;
qold=0;
Sy=Tmax;
Kq=(-Sy/(Ixx));
qb=zeros;
Ixx
for it=1:9
    qb(it)=qold+(-Sy/(Ixx))*ASol(it)*dx(it);
    fprintf(['qb ',num2str(it),',',num2str(it+1),'= ',num2str(qb(it)),' N/mm\n']);
    qold=qb(it);
end
   
%% calcolo delle aree di Bredt

%area di bredt 2
AB2=0;
AB2=2*trapz(solettex,solettey)*1000^2;

%matrici per wb
c=1.29;
xwb=[0 0.5 0.75 1.25 2.5 5.0 7.5 10 15 20 25]*c/100;
ywb=[0 0.973 1.173 1.492 2.078 2.895 3.504 3.994 4.747 5.287 5.664].*c/100;

%area di bredt 1
AB1=0;
AB1=2*trapz(xwb,ywb)*1000^2;

%% calcolo distanze
%area2
x=dy;
y=dx;
dist=zeros;
it=0;
for it=2:10
    dist(it)=sqrt(((x(it)-x(it-1)).^2)+((y(it)-y(it-1)).^2));
    it=it+1;
end
it=0;

%bordo d'attacco area1
it=0;
for it=2:10
    distLe(it)=sqrt(((xwb(it)-xwb(it-1)).^2)+((ywb(it)-ywb(it-1)).^2));
    it=it+1;
end
distLe=distLe.*1000;
%% perimetri
%perimetro bordo d'attacco area1
perimetrocirc=0; 
for it=1:10
    perimetrocirc=perimetrocirc+distLe(it);
    it=it+1;
end
perimetrocirc=perimetrocirc*2;
 %perimetro area 2
perimetrowb=sum(dist,'all')+dist(6);

%% calcolo prima equazione
qk=0; it=0;
for it=1:9
    qk=qk+qb(it)*dist(it+1);
    it=it+1;
end


    coeff_qs01=-(perimetrocirc-dist(6)-dist(6));
    coeff_qs02=((perimetrowb+dist(6))/AB2)+(dist(6)/AB1);
    termine_noto_1_eq=-qk/AB2;

%% calcolo seconda equazione
%AB1*qs01+AB2*qs02=flussitot/2
%y=y+y(1);

flussitot=0;
for it=1:9
    flussitot=flussitot+(qb(it)*dist(it+1)*y(it));
    it=it+1;
end
termine_noto_2_eq=flussitot/2;

%% soluzione sistema lineare
Sistema=[coeff_qs01,coeff_qs02
        AB1,AB2];
Noti=[termine_noto_1_eq
    termine_noto_2_eq];
Soluzione=linsolve(Sistema,Noti);
qs0=Soluzione;
display(qs0);

%% calcolo qs finale
qs=zeros;
for it=1:9
    qs(it)=qb(it)+qs0(2);
end
qs(10)=qb(6)+qs0(2)-qs0(1);
qs(11)=qs0(1);
for it=1:9
fprintf(['qs ',num2str(it),',',num2str(it+1),'= ',num2str(qs(it)),'\n']);
it=it+1;
end
fprintf(['qs ',num2str(10),',1= ',num2str(qs(it)),'\n']);
fprintf(['qs,circ ',num2str(it),'= ',num2str(qs(11)),'\n']);
%qs max
qsmax=max(abs(qs));
taumax=qsmax/ta;
sigmaeq=sqrt((sigmamax)^2+3*(taumax)^2); %minore si sigmacy 450MPa
MS=(sigmacy/sigmaeq)-1;

fprintf(['\nqs,max: ',num2str(qsmax),'\n']);
fprintf(['taumax= ',num2str(taumax),'\n']);
fprintf(['sigmaeq= ',num2str(sigmaeq),'\n']);
fprintf(['M.S.= ',num2str(MS),'\n']);

%% -- Capitolo 5 --
%% Torsione - calcolo flussi di taglio nel punto C con n negativo
fprintf('\n\nTorsione-----------------\n\n');
%% Torsione - calcolo delle sigma
Mx=-Mmax_Torsione*1000;
sigmazeta_su_y=(Mx/(Ixx));
it=0;
sigmazeta(10)=zeros;
for it=1:10
sigmazeta(it)=sigmazeta_su_y*dx(it);
fprintf(['Sigma ',num2str(it),'= ',num2str(sigmazeta(it)),'MPa\n']);
it=it+1;
end

%% Torsione - calcolo qb
it=0;
qold=0;
Sy=Tmax_Torsione;
Kq=(-Sy/(Ixx));
qb=zeros;
for it=1:9
    qb(it)=qold+(-Sy/(Ixx))*ASol(it)*dx(it);
    fprintf(['qb ',num2str(it),',',num2str(it+1),'= ',num2str(qb(it)),' N/mm\n']);
    qold=qb(it);
end

%% Torsione - calcolo prima equazione
qk=0; it=0;
for it=1:9
    qk=qk+qb(it)*dist(it+1);
    it=it+1;
end


    coeff_qs01=-(perimetrocirc-dist(6)-dist(6));
    coeff_qs02=((perimetrowb+dist(6))/AB2)+(dist(6)/AB1);
    termine_noto_1_eq=-qk/AB2;

%% Torsione -  calcolo seconda equazione
%AB1*qs01+AB2*qs02=flussitot/2

termine_noto_2_eq=(Mx)/2;

%%  Torsione - soluzione sistema lineare
Sistema=[coeff_qs01,coeff_qs02
        AB1,AB2];
Noti=[termine_noto_1_eq
    termine_noto_2_eq];
Soluzione=linsolve(Sistema,Noti);
qs0=Soluzione;
display(qs0);
%% Torsione - calcolo qs finale
qs=zeros;
for it=1:9
    qs(it)=qb(it)+qs0(2);
end
qs(10)=qb(6)+qs0(2)-qs0(1);
qs(11)=qs0(1);
for it=1:9
fprintf(['qs ',num2str(it),',',num2str(it+1),'= ',num2str(qs(it)),'\n']);
it=it+1;
end
fprintf(['qs ',num2str(10),',1= ',num2str(qs(it)),'\n']);
fprintf(['qs,circ ',num2str(it),'= ',num2str(qs(11)),'\n']);
%qs max
qsmax=max(abs(qs));
taumax=qsmax/ta;
sigmaeq=sqrt((sigmamax)^2+3*(taumax)^2); %minore si sigmacy 450MPa
MS=(sigmacy/sigmaeq)-1;

fprintf(['\nqs,max: ',num2str(qsmax),'\n']);
fprintf(['taumax= ',num2str(taumax),'\n']);
fprintf(['sigmaeq= ',num2str(sigmaeq),'\n']);
fprintf(['M.S.= ',num2str(MS),'\n']);

%% -- Capitolo 6 --
%% Instabilità del controvento
L=2100*cos(0.87); %controvento %angolo 50°
E=717000;
sigmacy=485;
sigma07=490;
sigma085=465;
sigmap=220;
r=0.03*1000;
%Area=pi*r^2;
Area=185.35;
Imin=80000;
rhomin=sqrt(Imin/Area);
lambda=L/rhomin;
lambdalim=pi*sqrt(E/sigmap);

sigmaeulero=(((pi^2))/(lambda^2))*E;
n=1+(log(17/7))/(log(sigma07/sigma085));
sigma=0;
figure(14)
xline(E,'-.k'); hold on;
i=1;
while sigma<sigmaeulero
    i=i+1;
Et=E/(1+(3/7)*n*(sigma/sigma07)^(n-1));
sigma=sigma+0.3;
Et_matrix(i)=Et;
sigma_matrix(i)=sigma;
end
%trim del primo valore
Et_matrix=Et_matrix(2:end);
sigma_matrix=sigma_matrix(2:end);
%grafica
verde=plot(Et_matrix,sigma_matrix, 'g'); hold on;

rosso=fplot(@(E) (((pi^2))/(lambda^2))*E,[0 E],'r');


yline(sigmaeulero,'--b'); hold on;
xlabel('Et(MPa)');
ylabel('\sigmacr(MPa)');
grid on; grid minor;

%Et 198065
sigmacr=460.13; %Mpa
Pcr=sigmacr*Area;
display(Pcr);
