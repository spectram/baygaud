#!/bin/csh -f

#mpif90 -O3 -DMPI -ffree-line-length-none -fPIC -c -o kmeans_clstr.o kmeans_clstr.f90
#mpif90 -O3 -DMPI -ffree-line-length-none -fPIC -c -o priors.o priors.f90
#mpif90 -O3 -DMPI -ffree-line-length-none -fPIC -c -o utils1.o utils1.f90
#mpif90 -O3 -DMPI -ffree-line-length-none -fPIC -c -o utils.o utils.f90
#mpif90 -O3 -DMPI -ffree-line-length-none -fPIC -c -o xmeans_clstr.o xmeans_clstr.f90
#mpif90 -O3 -DMPI -ffree-line-length-none -fPIC -c -o nested.o nested.F90
#mpif90 -O3 -DMPI -ffree-line-length-none -fPIC -c -o posterior.o posterior.F90
#ld -shared /usr/lib/liblapack.so.3 -lpthread -o libnest3.so utils.o utils1.o priors.o kmeans_clstr.o xmeans_clstr.o posterior.o nested.o

gfortran -O3 -ffree-line-length-none -fPIC -c -o kmeans_clstr_mpioff.o kmeans_clstr_mpioff.f90
gfortran -O3 -ffree-line-length-none -fPIC -c -o priors_mpioff.o priors_mpioff.f90
gfortran -O3 -ffree-line-length-none -fPIC -c -o utils1_mpioff.o utils1_mpioff.f90
gfortran -O3 -ffree-line-length-none -fPIC -c -o utils_mpioff.o utils_mpioff.f90
gfortran -O3 -ffree-line-length-none -fPIC -c -o xmeans_clstr_mpioff.o xmeans_clstr_mpioff.f90
gfortran -O3 -ffree-line-length-none -fPIC -c -o nested_mpioff.o nested_mpioff.F90
gfortran -O3 -ffree-line-length-none -fPIC -c -o posterior_mpioff.o posterior_mpioff.F90

#rm -rf kmeans_clstr_mpioff.mod
#rm -rf nested_mpioff.mod
#rm -rf posterior_mpioff.mod
#rm -rf priors_mpioff.mod
#rm -rf randomns_mpioff.mod
#rm -rf utils1_mpioff.mod
#rm -rf xmeans_clstr_mpioff.mod

#cp modules (*.mod) from MultiNest_v2.18
#cp ../MultiNest_v3.7/kmeans_clstr.mod kmeans_clstr_mpioff.mod
#cp ../MultiNest_v3.7/nested.mod nested_mpioff.mod 
#cp ../MultiNest_v3.7/posterior.mod posterior_mpioff.mod
#cp ../MultiNest_v3.7/priors.mod  priors_mpioff.mod
#cp ../MultiNest_v3.7/randomns.mod randomns_mpioff.mod
#cp ../MultiNest_v3.7/utils1.mod utils1_mpioff.mod
#cp ../MultiNest_v3.7/xmeans_clstr.mod xmeans_clstr_mpioff.mod

ld -shared /usr/lib/x86_64-linux-gnu/atlas/liblapack.so.3 -lpthread -o libnest3_mpioff.so utils_mpioff.o utils1_mpioff.o priors_mpioff.o kmeans_clstr_mpioff.o xmeans_clstr_mpioff.o posterior_mpioff.o nested_mpioff.o

ar cr libnest3_mpioff.a utils_mpioff.o utils1_mpioff.o priors_mpioff.o kmeans_clstr_mpioff.o xmeans_clstr_mpioff.o posterior_mpioff.o nested_mpioff.o


# locate libnest3_mpioff.a
#sudo cp libnest3_mpioff.a /usr/local/lib/libnest3_mpioff.a

# locate libnest3_mpioff.so
#sudo cp libnest3_mpioff.so /usr/lib/libnest3_mpioff.so


