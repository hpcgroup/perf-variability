Follow https://github.com/LLNL/mpiP for configuration. In the **generated** mpiP-wrappers.c, do following changes:

---

1.In function *static int mpiPif_MPI_Allgatherv(...)*

mpiPi_stats_mt_enter(hndl); \
++ if ( *sendtype == MPI_DATATYPE_NULL ) sendbuf = MPI_IN_PLACE; \
rc = PMPI_Allgather( sendbuf,  * sendcount, …)

2.In function *static int mpiPif_MPI_Allgather*

mpiPi_stats_mt_enter(hndl); \
++ if ( *sendtype == MPI_DATATYPE_NULL ) sendbuf = MPI_IN_PLACE; \
rc = PMPI_Allgather( sendbuf,  * sendcount, ...);

--- 

Then make and compile.